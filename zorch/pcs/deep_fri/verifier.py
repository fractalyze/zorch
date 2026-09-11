# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""DEEP-FRI verifier: rebuild the batched composition from the `f_m`, then
check fold consistency.

For each query the verifier reconstructs the batched DEEP composition at the
conjugate pair from the *committed* `f_m` values —
`g(x) = Σ_m vf^m · (f_m(x) − v_m)/(x − z_m)` — and binds that pair to the
committed layer-0 leaf, so a false claim `v_m ≠ f_m(z_m)` yields a
non-low-degree `g` that fails both the per-layer fold check and the final-layer
constant check. It never trusts a prover-sent layer-0 oracle. The rebuild
groups per column where the prover's `deep_composition` groups per opening
point; field arithmetic is exact, so the two orders agree bit-for-bit. All
arithmetic (NTT domain, field divide, Merkle rebuild) lowers on CPU and GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
from frx import Array, lax

from zorch.pcs.deep_fri.config import DeepFriCommitment, DeepFriParams, DeepFriProof
from zorch.pcs.fold import sample_positions, verify_fold_chain, verify_openings
from zorch.pcs.stage import OpeningClaim, OpeningProof
from zorch.poly.univariate import powers
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class DeepFriVerifier(
    VerifierStage[
        OpeningClaim[DeepFriCommitment],
        TrivialClaim,
        OpeningProof[DeepFriProof],
    ]
):
    params: DeepFriParams

    def verify(
        self,
        claim: OpeningClaim[DeepFriCommitment],
        reduction_proof: OpeningProof[DeepFriProof],
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        """Check the claimed evaluations against the commitments."""
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
        commitment: DeepFriCommitment,
        points: Sequence[Array],
        values: Array,
        proof: DeepFriProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == len(proof.f_openings) == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, f_openings={len(proof.f_openings)}"
            )
        # Fail loud on a structurally short proof: the replay scan iterates over
        # whatever fri_roots it is handed, so a missing layer would silently skip
        # a round's checks rather than error. Eager, ahead of the jit zone.
        rounds = self.params.num_rounds
        if len(proof.fri_roots) != rounds or len(proof.query_openings) != rounds:
            raise ValueError(
                f"malformed proof: expected {rounds} fold layers, got "
                f"{len(proof.fri_roots)} roots / "
                f"{len(proof.query_openings)} openings"
            )
        t, ok = _verify_batch(
            self.params, commitment, fnp.stack(points), values, proof, transcript
        )
        return ok, t


# Jitted whole-batch verify body; the params are the static key (by
# value, #214).
@partial(frx.jit, static_argnames=("params",))
def _verify_batch(
    params: DeepFriParams,
    roots: DeepFriCommitment,
    xis: Array,
    values: Array,
    pf: DeepFriProof,
    t: Transcript,
) -> tuple[Transcript, Array]:
    code, tree = params.code, params.tree
    n = code.block_len
    num_rounds = params.num_rounds
    m = roots.shape[0]

    # Replay Fiat-Shamir: the statement bind and batching-challenge squeeze
    # mirror the prover, then each round observes its pre-fold commitment root
    # before sampling β — all num_rounds rounds are homogeneous and ride one
    # lax.scan (a stablehlo.while) — then the cleartext final layer.
    t = t.observe(roots).observe(xis)
    t, vf = t.observe_and_sample(values)
    vf = vf.reshape(())

    def fold_round(t: Transcript, root: Array) -> tuple[Transcript, Array]:
        t = t.observe(root)
        t, beta = t.sample()
        return t, beta.reshape(())

    t, betas_stacked = lax.scan(fold_round, t, fnp.stack(pf.fri_roots))
    # Index, don't iterate: list(field_array) dispatches lax.sign under CUDA.
    betas = [betas_stacked[r] for r in range(num_rounds)]
    t = t.observe(pf.final_layer)

    # The final fold layer must be a constant (degree 0). DEEP-FRI binds no
    # external claim, so the layer's own head is the claimed message.
    ok = code.check_final(pf.final_layer, pf.final_layer[0])

    t, positions = sample_positions(t, n, params.num_queries)
    a = code.layer_positions(positions, num_rounds)

    # Merkle: each f_m's pair-leaf rebuilds its root at a[0]; each fold layer's
    # pair-leaf rebuilds its committed root at a[i]. One batched pass (#163).
    legs = [(roots[i], a[0], pf.f_openings[i]) for i in range(m)]
    for i in range(num_rounds):
        legs.append((pf.fri_roots[i], a[i], pf.query_openings[i]))
    ok = ok & verify_openings(tree, legs)

    # Rebuild the layer-0 composition pair from the opened f_m leaves and bind
    # it to the committed layer-0 pair — this is what ties the fold chain to
    # the f_m.
    lo0, hi0 = code.pair_indices(a[0], 0)
    domain = code.domain()
    x_lo, x_hi = domain[lo0], domain[hi0]
    vf_pows = powers(vf, m)
    g_lo, g_hi = fnp.zeros_like(x_lo), fnp.zeros_like(x_hi)
    for i in range(m):
        numer_lo = vf_pows[i] * (pf.f_openings[i].row[:, 0] - values[i])
        numer_hi = vf_pows[i] * (pf.f_openings[i].row[:, 1] - values[i])
        g_lo = g_lo + numer_lo / (x_lo - xis[i])
        g_hi = g_hi + numer_hi / (x_hi - xis[i])
    layer0 = pf.query_openings[0].row  # (Q, 2)
    ok = ok & fnp.all(g_lo == layer0[:, 0]) & fnp.all(g_hi == layer0[:, 1])

    # Each layer's opened pair folds to the next layer's / the final layer.
    ok = ok & verify_fold_chain(code, pf.query_openings, betas, a, pf.final_layer)
    return t, ok


if TYPE_CHECKING:
    _: type[
        VerifierStage[
            OpeningClaim[DeepFriCommitment],
            TrivialClaim,
            OpeningProof[DeepFriProof],
        ]
    ] = DeepFriVerifier
