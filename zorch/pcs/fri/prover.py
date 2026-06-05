# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI prover for univariate point evaluation (the DEEP quotient trick).

To open `f` at `z` with claim `v`, the prover shows the quotient
`g(x) = (f(x) − v)/(x − z)` is low degree: `g` is a polynomial of degree
`deg f − 1` exactly when `v = f(z)`. `g` is never committed separately — the
verifier rebuilds its codeword from the already-committed `f` at queried points —
so `open` Merkle-commits only the folded layers and threads the transcript through
the fold challenges. This is the structural opposite of KZG on the same PCS seam:
interactive, Merkle-backed, no SRS, and entirely field/NTT arithmetic that lowers
on both CPU and GPU (no MSM). The query phase is device-batched — positions are a
device int32 array and each Merkle opening is one `vmap` over them, no host indices
or per-query loop — while the fold phase stays a Python `for` over rounds: each
round Merkle-commits a half-size layer, so the loop is not `scan`-shaped (see
docs/conventions.md). Scope: a single base-field polynomial per opening, fixed
small parameters — a demonstration of the seam, not a hardened prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from zorch.pcs.fold import CommitFoldRound, open_query_phase
from zorch.pcs.fri.config import FriCommitment, FriParams, FriProof
from zorch.prove import fold_rounds
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


def _eval_poly(coeffs: Array, z: Array) -> Array:
    """f(z) for ascending `coeffs`, by index-loop Horner (iterating a field array
    under CUDA dispatches lax.sign — index instead)."""
    acc = jnp.zeros((), coeffs.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * z + coeffs[i]
    return acc


@dataclass(frozen=True)
class FriCommittedPoly:
    """One committed polynomial's retained witness: ascending coefficients, the
    `[n, 1]` codeword matrix (the Merkle leaves), and its digest layers."""

    coeffs: Array
    matrix: Array
    digest_layers: list[Array]


@dataclass(frozen=True)
class FriProverData:
    """Retained witness from `FriProver.commit`, one entry per committed poly."""

    polys: tuple[FriCommittedPoly, ...]


@dataclass(frozen=True)
class FriProver:
    params: FriParams

    def commit(self, polys: Sequence[Array]) -> tuple[FriCommitment, FriProverData]:
        """RS-encode each coefficient vector and Merkle-commit the codeword.
        Returns stacked roots and the prover data needed to open."""
        roots, committed = [], []
        for coeffs in polys:
            codeword = self.params.code.encode(coeffs)
            matrix = codeword.reshape(-1, 1)
            root, digest_layers = self.params.tree.commit(matrix)
            roots.append(root)
            committed.append(FriCommittedPoly(coeffs, matrix, digest_layers))
        return jnp.stack(roots), FriProverData(tuple(committed))

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
            t, proof = self._open_one(committed, z, v, t)
            values.append(v)
            proofs.append(proof)
        return jnp.stack(values), proofs, t

    def _open_one(
        self,
        committed: FriCommittedPoly,
        z: Array,
        v: Array,
        t: Transcript,
    ) -> tuple[Transcript, FriProof]:
        matrix, digest_layers = committed.matrix, committed.digest_layers
        params = self.params
        domain = params.code.domain()
        f_codeword = matrix.reshape(-1)
        quotient = (f_codeword - v) / (domain - z)  # layer 0, rebuilt from f at verify

        t = t.observe(digest_layers[-1][0])  # bind the f commitment root
        cw, t, layers = fold_rounds(
            CommitFoldRound(params.code, params.tree),
            quotient,
            t,
            params.num_rounds - 1,
        )
        # Final round, peeled: the fully folded layer goes in the clear, no commit.
        t, beta = t.sample()
        final_layer = params.code.fold(cw, beta.reshape(()))
        t = t.observe(final_layer)

        t, base, layer_opens = open_query_phase(
            params.tree, t, matrix, digest_layers, layers, params.num_queries
        )
        layer_roots = [layer.root for layer in layers]
        return t, FriProof(v, layer_roots, final_layer, base.lo, base.hi, layer_opens)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsProver[FriCommitment, FriProverData, list[FriProof]]] = FriProver
