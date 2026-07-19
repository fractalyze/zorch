# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged LogUp-GKR verifier -- the dual of the jagged prover chain.

A `JaggedGkrLayerRound` replays one layer's coefficient-form sumcheck (the
agnostic `zorch.verify` driver over `sumcheck.verifier.CoeffsSumcheckRound`),
checks the LogUp oracle at the bound point via the shared `logup_combine`,
then reduces the claim across the child selector -- the same
`(num_eval, den_eval, eval_point)` carry the dense chain threads. The jagged
layout never reaches the verifier: the prover's virtual-mass corrections make
its round polynomials exactly those of the virtual dense hypercube, so the
verifier is layout-blind and stays succinct (`eval_eq`, no 2^n vector).

The prover binds LSB-first, so the bound point is the sampled challenges
reversed -- the one place the jagged dual differs from the dense verifier,
whose MSB-first prover emits challenges already in point order.

It stops at the reduced point-claim. The final `claim == leaf_mle(point)`
check needs a PCS opening of the input trace and is the consumer's, keeping
this block PCS-agnostic; the roundtrip tests close it directly against the
virtual dense leaf MLE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array

from zorch.logup_gkr.jagged_prover import JaggedLayerProof
from zorch.logup_gkr.prover import Carry, fold_carry, logup_combine
from zorch.poly.eq import eval_eq
from zorch.round import Round
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.transcript import Transcript, sample_challenge
from zorch.verify import verify

if TYPE_CHECKING:
    from zorch.round import VerifierRound

_DEGREE = 3  # LogUp combine round-polynomial degree (eq * deg-2 bracket).


class JaggedGkrLayerRound(Round):
    """Verify one jagged GKR layer; the chain of these is the jagged GKR
    verifier. `challenge_limbs` must match the prover's -- every challenge in
    the layer comes from the same squeeze rule."""

    def __init__(self, challenge_limbs: int = 1) -> None:
        self.challenge_limbs = challenge_limbs

    def __call__(
        self, carry: Carry, layer_proof: JaggedLayerProof, transcript: Transcript
    ) -> tuple[Carry, Transcript, Array]:
        num_eval, den_eval, eval_point = carry
        n0, n1 = layer_proof.numerator_0, layer_proof.numerator_1
        d0, d1 = layer_proof.denominator_0, layer_proof.denominator_1
        transcript, lam = sample_challenge(
            transcript, num_eval.dtype, self.challenge_limbs
        )
        claim = lam * num_eval + den_eval
        point, final_claim, transcript, ok_sc = verify(
            CoeffsSumcheckRound(_DEGREE, self.challenge_limbs),
            claim,
            layer_proof.round_polys,
            transcript,
        )
        # LSB-first binding: the last challenge bound the MSB, so the
        # MSB-first bound point is the sample order reversed.
        point = point[::-1]
        # The carry's eval_point must have one coordinate per sumcheck round, or
        # `eval_eq` reads the wrong eq weight (a degenerate length-1 carry would
        # broadcast silently against the bound point) -- reject, never broadcast.
        if eval_point.shape[0] != point.shape[0]:
            raise ValueError(
                f"eq point mismatch: carry has {eval_point.shape[0]} coords, "
                f"layer ran {point.shape[0]} rounds"
            )
        # LogUp oracle: the reduced claim equals the combine at the bound
        # point, with eq evaluated in closed form (both points MSB-first).
        eq_eval = eval_eq(eval_point, point)
        combined = logup_combine(lam, eq_eval, n0, d1, n1, d0)
        ok = ok_sc & (combined == final_claim)

        transcript = transcript.observe(fnp.stack([n0, n1, d0, d1]))
        transcript, r = sample_challenge(
            transcript, num_eval.dtype, self.challenge_limbs
        )
        num_eval, den_eval, eval_point = fold_carry(n0, n1, d0, d1, point, r)
        return (num_eval, den_eval, eval_point), transcript, ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[VerifierRound] = JaggedGkrLayerRound
