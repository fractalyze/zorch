# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI prover for univariate point evaluation (the DEEP quotient trick).

To open `f` at `z` with claim `v`, the prover shows the quotient
`g(x) = (f(x) − v)/(x − z)` is low degree: `g` is a polynomial of degree
`deg f − 1` exactly when `v = f(z)`. `g` is never opened from a separate
commitment — the verifier rebuilds its layer-0 pair from the already-committed
`f` at the queried points and checks it against the committed fold chain — so
`open` Merkle-commits `g`'s fold layers (conjugate-pair leaves, pre-fold) and
threads the transcript through the fold challenges. This is the structural
opposite of KZG on the same PCS seam: interactive, Merkle-backed, no SRS, and
entirely field/NTT arithmetic that lowers on both CPU and GPU (no MSM). The query
phase is device-batched — positions are a device int32 array and each Merkle
opening is one `vmap` over them — while the fold phase stays a Python `for` over
rounds: each round Merkle-commits a half-size layer, so the loop is not
`scan`-shaped (see docs/reference/conventions.md). Scope: a single base-field polynomial
per opening, fixed small parameters — a demonstration of the seam, not a hardened
prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
from frx import Array

from zorch.commit.merkle import MerkleTree
from zorch.pcs.fold import (
    FoldState,
    PreFoldPairCommitRound,
    open_rows,
    sample_positions,
)
from zorch.pcs.fri.config import DeepFoldableCode, FriCommitment, FriParams, FriProof
from zorch.prove import fold_rounds
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


def _eval_poly(coeffs: Array, z: Array) -> Array:
    """f(z) for ascending `coeffs`, by index-loop Horner (iterating a field array
    under CUDA dispatches lax.sign — index instead)."""
    acc = fnp.zeros((), coeffs.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * z + coeffs[i]
    return acc


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["coeffs", "codeword", "leaves", "digest_layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class FriCommittedPoly:
    """One committed polynomial's retained witness: ascending coefficients, the
    raw `[n]` codeword (the DEEP quotient divides it pointwise), the `[n//2, 2]`
    conjugate-pair leaves (the Merkle commitment), and its digest layers.

    A registered pytree so it crosses the `open` `@jit` boundary."""

    coeffs: Array
    codeword: Array
    leaves: Array
    digest_layers: list[Array]


@dataclass(frozen=True)
class FriProverData:
    """Retained witness from `FriProver.commit`, one entry per committed poly."""

    polys: tuple[FriCommittedPoly, ...]


@dataclass(frozen=True)
class FriProver:
    params: FriParams

    def commit(self, polys: Sequence[Array]) -> tuple[FriCommitment, FriProverData]:
        """RS-encode each coefficient vector and Merkle-commit its codeword's
        conjugate-pair leaves. Returns stacked roots and the prover data."""
        committed = [
            _commit_one(self.params.code, self.params.tree, coeffs) for coeffs in polys
        ]
        roots = [poly.digest_layers[-1][0] for poly in committed]
        return fnp.stack(roots), FriProverData(tuple(committed))

    def open(
        self,
        prover_data: FriProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, list[FriProof], Transcript]:
        if len(prover_data.polys) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data.polys)} polys vs "
                f"{len(points)} points"
            )
        values, proofs = [], []
        t = transcript
        for committed, z in zip(prover_data.polys, points):
            v = _eval_poly(committed.coeffs, z)
            t, proof = _open_one(self.params, committed, z, v, t)
            values.append(v)
            proofs.append(proof)
        return fnp.stack(values), proofs, t


# Jitted per-poly commit/open bodies, like basefold's zones; one
# compile serves the batch. Commit is keyed on code + tree, not the whole
# FriParams: it never reads the open-side knobs (num_rounds / num_queries), so
# params differing only there must not compile twice (static keys compare by
# value — #214).
@partial(frx.jit, static_argnames=("code", "tree"))
def _commit_one(
    code: DeepFoldableCode, tree: MerkleTree, coeffs: Array
) -> FriCommittedPoly:
    codeword = code.encode(coeffs)
    leaves = code.pair_leaves(codeword)
    _root, digest_layers = tree.commit(leaves)
    return FriCommittedPoly(coeffs, codeword, leaves, digest_layers)


@partial(frx.jit, static_argnames=("params",))
def _open_one(
    params: FriParams,
    committed: FriCommittedPoly,
    z: Array,
    v: Array,
    t: Transcript,
) -> tuple[Transcript, FriProof]:
    code, tree = params.code, params.tree
    domain = code.domain()
    # Layer 0 is the DEEP quotient, derived from f; the round commits it
    # before folding (like every layer), and the verifier rebuilds this same
    # pair from f's opened leaf to bind the commitment chain to f.
    quotient = (committed.codeword - v) / (domain - z)

    t = t.observe(committed.digest_layers[-1][0])  # bind the f commitment root
    state, t, roots = fold_rounds(
        PreFoldPairCommitRound(code, tree), FoldState(quotient), t, params.num_rounds
    )
    final_layer = state.codeword
    t = t.observe(final_layer)

    n = code.block_len
    t, positions = sample_positions(t, n, params.num_queries)
    a = code.layer_positions(positions, params.num_rounds)
    f_opening = open_rows(tree, committed.leaves, committed.digest_layers, a[0])
    query_openings = [
        open_rows(tree, layer.leaves, layer.digest_layers, a[i])
        for i, layer in enumerate(state.layers)
    ]
    return t, FriProof(
        v,
        roots,
        final_layer,
        f_opening,
        query_openings,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[PcsProver[FriCommitment, FriProverData, list[FriProof]]] = FriProver
