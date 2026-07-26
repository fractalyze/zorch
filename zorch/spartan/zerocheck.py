# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Outer Spartan zerocheck role implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.poly.eq import eval_eq
from zorch.poly.multilinear import eval_mle
from zorch.spartan.summand import ZerocheckSummand
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.sumcheck.eq.stage import (
    EqPolyProver,
    EqPolyVerifier,
    EqPolyWitness,
    EqSumClaim,
)
from zorch.sumcheck.stage import EvaluationClaim
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class ZerocheckClaim:
    """Public claim that the outer relation vanishes over this many variables."""

    rounds: int


@dataclass(frozen=True)
class ZerocheckWitness:
    """The three multilinear factor tables witnessing ``ZerocheckClaim``."""

    az: Array
    bz: Array
    cz: Array


@dataclass(frozen=True)
class RowEvaluationClaim:
    """Claimed ``(Az, Bz, Cz)`` evaluations at the reduced row point."""

    point: Array
    values: Array


@dataclass(frozen=True)
class OuterProof:
    """The outer sumcheck messages and terminal ``(Az, Bz, Cz)`` values."""

    sumcheck: Any
    claims: Array


def _sample_tau(
    rounds: int, transcript: Transcript, challenges: ChallengePolicy
) -> tuple[Array, Transcript]:
    values = []
    for _ in range(rounds):
        transcript, challenge = challenges.sample(transcript)
        values.append(challenge)
    return fnp.stack(values), transcript


class OuterProver(
    ProverStage[ZerocheckClaim, ZerocheckWitness, RowEvaluationClaim, OuterProof]
):
    """Prove zerocheck conditional on a row-evaluation claim."""

    def __init__(
        self,
        *,
        sumcheck: (
            ProverStage[EqSumClaim, EqPolyWitness, EvaluationClaim, Any] | None
        ) = None,
        challenges: ChallengePolicy,
    ) -> None:
        self.challenges = challenges
        self.sumcheck = sumcheck or EqPolyProver(
            ZerocheckSummand(), challenges=challenges
        )

    def prove(
        self,
        claim: ZerocheckClaim,
        witness: ZerocheckWitness,
        transcript: Transcript,
    ) -> ProveResult[RowEvaluationClaim, OuterProof]:
        az, bz, cz = witness.az, witness.bz, witness.cz
        rounds = log2_strict_usize(az.shape[0])
        if rounds != claim.rounds:
            raise ValueError(
                f"claim expects {claim.rounds} outer rounds, witness needs {rounds}"
            )
        tau, transcript = _sample_tau(claim.rounds, transcript, self.challenges)
        factors = fnp.stack([az, bz, cz])
        zero = fnp.zeros((), tau.dtype)
        reduced = self.sumcheck.prove(
            EqSumClaim(tau, zero, claim.rounds),
            EqPolyWitness(factors),
            transcript,
        )
        point = reduced.reduced_claim.point
        values = fnp.stack(
            [eval_mle(az, point), eval_mle(bz, point), eval_mle(cz, point)]
        )
        transcript = reduced.transcript.observe(values)
        return ProveResult(
            RowEvaluationClaim(point, values),
            OuterProof(reduced.reduction_proof, values),
            transcript,
        )


class OuterVerifier(VerifierStage[ZerocheckClaim, RowEvaluationClaim, OuterProof]):
    """Verify zerocheck conditional on a row-evaluation claim."""

    def __init__(
        self,
        *,
        sumcheck: VerifierStage[EqSumClaim, EvaluationClaim, Any] | None = None,
        challenges: ChallengePolicy,
    ) -> None:
        self.challenges = challenges
        self.sumcheck = sumcheck or EqPolyVerifier(
            ZerocheckSummand(), challenges=challenges
        )

    def verify(
        self,
        claim: ZerocheckClaim,
        reduction_proof: OuterProof,
        transcript: Transcript,
    ) -> VerifyResult[RowEvaluationClaim]:
        tau, transcript = _sample_tau(claim.rounds, transcript, self.challenges)
        # Both roles derive the source-claim field from the public challenge.
        zero = fnp.zeros((), tau.dtype)
        reduced = self.sumcheck.verify(
            EqSumClaim(tau, zero, claim.rounds),
            reduction_proof.sumcheck,
            transcript,
        )
        point = reduced.reduced_claim.point
        final_value = reduced.reduced_claim.value
        transcript = reduced.transcript.observe(reduction_proof.claims)
        va, vb, vc = reduction_proof.claims
        ok = reduced.ok & (final_value == eval_eq(tau, point) * (va * vb - vc))
        return VerifyResult(
            RowEvaluationClaim(point, reduction_proof.claims), transcript, ok
        )
