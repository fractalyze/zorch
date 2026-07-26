# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI verifier: rebuild the quotient from `f`, then check fold consistency.

For each query the verifier reconstructs the quotient `g` at the conjugate pair
from the *committed* `f` values — `g(x) = (f(x) − v)/(x − z)` — and binds that
pair to the committed layer-0 leaf, so a false claim `v ≠ f(z)` yields a
non-low-degree `g` that fails both the per-layer fold check and the final-layer
constant check. It never trusts a prover-sent layer-0 oracle. All arithmetic
(NTT domain, field divide, Merkle rebuild) lowers on CPU and GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
from frx import Array, lax

from zorch.pcs.fold import sample_positions, verify_fold_chain, verify_openings
from zorch.pcs.fri.config import FriCommitment, FriParams, FriProof
from zorch.pcs.stage import OpeningClaim, OpeningProof
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class FriVerifier(
    VerifierStage[
        OpeningClaim[FriCommitment],
        TrivialClaim,
        OpeningProof[list[FriProof]],
    ]
):
    params: FriParams

    def verify(
        self,
        claim: OpeningClaim[FriCommitment],
        reduction_proof: OpeningProof[list[FriProof]],
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        """Check the claimed evaluations against the commitment."""
        ok, transcript = self._verify_opening(
            claim.commitment,
            claim.points,
            reduction_proof.values,
            reduction_proof.proof,
            transcript,
        )
        return VerifyResult(TrivialClaim(), transcript, ok)

    def _verify_opening(
        self,
        commitment: FriCommitment,
        points: Sequence[Array],
        values: Array,
        proof: Sequence[FriProof],
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == len(proof) == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, proof={len(proof)}"
            )
        # Fail loud on a structurally short proof: the replay scan iterates over
        # whatever fri_roots it is handed, so a missing layer would silently skip
        # a round's checks rather than error. Eager, ahead of the jit zone.
        rounds = self.params.num_rounds
        for pf in proof:
            if len(pf.fri_roots) != rounds or len(pf.query_openings) != rounds:
                raise ValueError(
                    f"malformed proof: expected {rounds} fold layers, got "
                    f"{len(pf.fri_roots)} roots / {len(pf.query_openings)} openings"
                )
        t = transcript
        oks = []
        for f_root, z, v, pf in zip(commitment, points, values, proof):
            t, ok = _verify_one(self.params, f_root, z, v, pf, t)
            oks.append(ok)
        return fnp.all(fnp.stack(oks)), t


# Jitted per-poly verify body; one compile serves the batch. The
# params are the static key (by value, #214).
@partial(frx.jit, static_argnames=("params",))
def _verify_one(
    params: FriParams, f_root: Array, z: Array, v: Array, pf: FriProof, t: Transcript
) -> tuple[Transcript, Array]:
    code, tree = params.code, params.tree
    n = code.block_len
    num_rounds = params.num_rounds

    # Replay Fiat-Shamir: each round observes its pre-fold commitment root
    # before sampling β, so all num_rounds rounds are homogeneous and ride
    # one lax.scan (a stablehlo.while), then the cleartext final layer.
    t = t.observe(f_root)

    def fold_round(t: Transcript, root: Array) -> tuple[Transcript, Array]:
        t = t.observe(root)
        t, beta = t.sample()
        return t, beta.reshape(())

    t, betas_stacked = lax.scan(fold_round, t, fnp.stack(pf.fri_roots))
    # Index, don't iterate: list(field_array) dispatches lax.sign under CUDA.
    betas = [betas_stacked[r] for r in range(num_rounds)]
    t = t.observe(pf.final_layer)

    # The final fold layer must be a constant (degree 0). FRI binds no
    # external claim, so the layer's own head is the claimed message.
    ok = code.check_final(pf.final_layer, pf.final_layer[0])

    t, positions = sample_positions(t, n, params.num_queries)
    a = code.layer_positions(positions, num_rounds)

    # Merkle: f's pair-leaf rebuilds f's root at a[0]; each fold layer's
    # pair-leaf rebuilds its committed root at a[i]. One batched pass (#163).
    legs = [(f_root, a[0], pf.f_opening)]
    for i in range(num_rounds):
        legs.append((pf.fri_roots[i], a[i], pf.query_openings[i]))
    ok = ok & verify_openings(tree, legs)

    # Rebuild the layer-0 quotient pair from f's opened leaf and bind it to
    # the committed layer-0 pair — this is what ties the fold chain to f.
    lo0, hi0 = code.pair_indices(a[0], 0)
    domain = code.domain()
    f_lo = pf.f_opening.row[:, 0]
    f_hi = pf.f_opening.row[:, 1]
    g_lo = (f_lo - v) / (domain[lo0] - z)
    g_hi = (f_hi - v) / (domain[hi0] - z)
    layer0 = pf.query_openings[0].row  # (Q, 2)
    ok = ok & fnp.all(g_lo == layer0[:, 0]) & fnp.all(g_hi == layer0[:, 1])

    # Each layer's opened pair folds to the next layer's / the final layer.
    ok = ok & verify_fold_chain(code, pf.query_openings, betas, a, pf.final_layer)
    return t, ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[
        VerifierStage[
            OpeningClaim[FriCommitment],
            TrivialClaim,
            OpeningProof[list[FriProof]],
        ]
    ] = FriVerifier
