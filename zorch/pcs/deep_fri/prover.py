# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""DEEP-FRI prover: one batched DEEP composition, one fold chain.

To open `M` committed polynomials, each `f_m` at its point `z_m` with claim
`v_m`, the prover batches the `M` DEEP quotients `(f_m(x) − v_m)/(x − z_m)` by
powers of one Fiat-Shamir challenge into a single codeword
(`zorch.pcs.deep.deep_composition`) — a genuine low-degree polynomial exactly
when every claim holds — and shows *that* is low degree with one commit-fold
chain and one query phase, instead of a chain per polynomial. The composition
is never opened from a separate commitment: the verifier rebuilds its layer-0
pair from the already-committed `f_m` values at the queried points and checks
it against the committed fold chain, so `open` Merkle-commits only the
composition's fold layers (conjugate-pair leaves, pre-fold) and threads the
transcript through the fold challenges. This is the structural opposite of KZG
on the same PCS seam: interactive, Merkle-backed, no SRS, and entirely
field/NTT arithmetic that lowers on both CPU and GPU (no MSM). The query phase
is device-batched — positions are a device int32 array and each Merkle opening
is one `vmap` over them — while the fold phase stays a Python `for` over
rounds: each round Merkle-commits a half-size layer, so the loop is not
`scan`-shaped (see docs/reference/conventions.md). Scope: base-field
polynomials, fixed small parameters — a demonstration of the seam, not a
hardened prover.
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
from zorch.pcs.deep import deep_composition
from zorch.pcs.deep_fri.config import (
    DeepFoldableCode,
    DeepFriCommitment,
    DeepFriParams,
    DeepFriProof,
)
from zorch.pcs.fold import (
    FoldState,
    PreFoldPairCommitRound,
    open_rows,
    sample_positions,
)
from zorch.pcs.stage import OpeningClaim, OpeningProof, OpeningWitness
from zorch.prove import fold_rounds
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import Transcript


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
class DeepFriCommittedPoly:
    """One committed polynomial's retained witness: ascending coefficients, the
    raw `[n]` codeword (the DEEP quotient divides it pointwise), the `[n//2, 2]`
    conjugate-pair leaves (the Merkle commitment), and its digest layers.

    A registered pytree so it crosses the `open` `@jit` boundary."""

    coeffs: Array
    codeword: Array
    leaves: Array
    digest_layers: list[Array]


@dataclass(frozen=True)
class DeepFriProverData:
    """Retained witness from `DeepFriProver.commit`, one entry per committed
    poly."""

    polys: tuple[DeepFriCommittedPoly, ...]


@dataclass(frozen=True)
class DeepFriProver(
    ProverStage[
        OpeningClaim[DeepFriCommitment],
        OpeningWitness[DeepFriProverData],
        TrivialClaim,
        OpeningProof[DeepFriProof],
    ]
):
    params: DeepFriParams

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[DeepFriCommitment, DeepFriProverData]:
        """RS-encode each coefficient vector and Merkle-commit its codeword's
        conjugate-pair leaves. Returns stacked roots and the prover data."""
        committed = [
            _commit_one(self.params.code, self.params.tree, coeffs) for coeffs in polys
        ]
        roots = [poly.digest_layers[-1][0] for poly in committed]
        return fnp.stack(roots), DeepFriProverData(tuple(committed))

    def prove(
        self,
        claim: OpeningClaim[DeepFriCommitment],
        witness: OpeningWitness[DeepFriProverData],
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof[DeepFriProof]]:
        """Open the committed polynomials at the claim's points as one batched
        DEEP-FRI opening.

        Terminal: an opening closes its claim rather than reducing it."""
        values, proof, transcript = self._open(
            witness.prover_data, claim.points, transcript
        )
        return ProveResult(TrivialClaim(), OpeningProof(values, proof), transcript)

    def _open(
        self,
        prover_data: DeepFriProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, DeepFriProof, Transcript]:
        if len(prover_data.polys) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data.polys)} polys vs "
                f"{len(points)} points"
            )
        t, values, proof = _open_batch(
            self.params, prover_data.polys, fnp.stack(points), transcript
        )
        return values, proof, t


# Jitted per-poly commit body / whole-batch open body, like basefold's
# zones. Commit is keyed on code + tree, not the whole DeepFriParams: it never
# reads the open-side knobs (num_rounds / num_queries), so params differing only
# there must not compile twice (static keys compare by value — #214).
@partial(frx.jit, static_argnames=("code", "tree"))
def _commit_one(
    code: DeepFoldableCode, tree: MerkleTree, coeffs: Array
) -> DeepFriCommittedPoly:
    codeword = code.encode(coeffs)
    leaves = code.pair_leaves(codeword)
    _root, digest_layers = tree.commit(leaves)
    return DeepFriCommittedPoly(coeffs, codeword, leaves, digest_layers)


@partial(frx.jit, static_argnames=("params",))
def _open_batch(
    params: DeepFriParams,
    committed: tuple[DeepFriCommittedPoly, ...],
    xis: Array,
    t: Transcript,
) -> tuple[Transcript, Array, DeepFriProof]:
    code, tree = params.code, params.tree
    domain = code.domain()
    m = len(committed)

    # Bind the whole statement — commitments, points, claimed values, in that
    # order — before squeezing the batching challenge: a challenge sampled
    # earlier would let the prover pick values after seeing it.
    roots = fnp.stack([poly.digest_layers[-1][0] for poly in committed])
    values = fnp.stack(
        [_eval_poly(poly.coeffs, xis[i]) for i, poly in enumerate(committed)]
    )
    t = t.observe(roots).observe(xis)
    t, vf = t.observe_and_sample(values)

    # The batched DEEP composition, layer 0 of the single fold chain. This
    # scheme is base-field-scoped, so every column rides the extension slot and
    # the base block is zero-width (`deep_composition` reads only its width);
    # a mixed-field consumer splits its columns instead.
    cols = fnp.stack([poly.codeword for poly in committed], axis=1)  # (n, M)
    composition = deep_composition(
        cols[:, :0], cols, values, xis, tuple(range(m)), vf.reshape(()), domain
    )

    state, t, fri_roots = fold_rounds(
        PreFoldPairCommitRound(code, tree), FoldState(composition), t, params.num_rounds
    )
    final_layer = state.codeword
    t = t.observe(final_layer)

    t, positions = sample_positions(t, code.block_len, params.num_queries)
    a = code.layer_positions(positions, params.num_rounds)
    f_openings = [
        open_rows(tree, poly.leaves, poly.digest_layers, a[0]) for poly in committed
    ]
    query_openings = [
        open_rows(tree, layer.leaves, layer.digest_layers, a[i])
        for i, layer in enumerate(state.layers)
    ]
    return t, values, DeepFriProof(fri_roots, final_layer, f_openings, query_openings)


if TYPE_CHECKING:
    _: type[
        ProverStage[
            OpeningClaim[DeepFriCommitment],
            OpeningWitness[DeepFriProverData],
            TrivialClaim,
            OpeningProof[DeepFriProof],
        ]
    ] = DeepFriProver
