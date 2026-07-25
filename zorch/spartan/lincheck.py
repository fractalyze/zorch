# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Spartan claim batching and inner lincheck roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import RowEvaluationClaim
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.sumcheck.prover import ProductSummand, StandardRound
from zorch.sumcheck.stage import (
    EvaluationClaim,
    SumcheckProver,
    SumcheckVerifier,
    SumcheckWitness,
    SumClaim,
)
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript


@dataclass(frozen=True)
class BatchedClaims:
    """The RLC challenge and joint value derived between the two stages."""

    challenge: Array
    joint: Array


def _joint_claim(claims: Array, challenge: Array) -> Array:
    va, vb, vc = claims[0], claims[1], claims[2]
    return va + challenge * vb + challenge * challenge * vc


def batch_claims(
    claims: Array,
    transcript: Transcript,
    challenges: ChallengePolicy,
) -> tuple[BatchedClaims, Transcript]:
    """Sample the batching challenge and derive the joint value."""
    transcript, challenge = challenges.sample(transcript)
    return BatchedClaims(challenge, _joint_claim(claims, challenge)), transcript


@dataclass(frozen=True)
class LincheckClaim:
    """Public claim reduced by the inner linearization sumcheck."""

    instance: R1CS
    row: RowEvaluationClaim
    batch: BatchedClaims


@dataclass(frozen=True)
class LincheckWitness:
    """Private assignment witnessing a ``LincheckClaim``."""

    assignment: Array


@dataclass(frozen=True)
class ColumnEvaluationClaim:
    """Claimed matrix-times-assignment evaluation at the column point."""

    point: Array
    value: Array


@dataclass(frozen=True)
class InnerProof:
    """The inner sumcheck reduction proof."""

    sumcheck: Any


class InnerProver(
    ProverStage[LincheckClaim, LincheckWitness, ColumnEvaluationClaim, InnerProof]
):
    """Prove lincheck conditional on a column-evaluation claim."""

    def __init__(
        self,
        *,
        sumcheck: (
            ProverStage[SumClaim, SumcheckWitness, EvaluationClaim, Any] | None
        ) = None,
        challenges: ChallengePolicy,
    ) -> None:
        self.sumcheck = sumcheck or SumcheckProver(
            StandardRound(ProductSummand(2), challenges=challenges),
            SumcheckRound(2, challenges),
        )

    def prove(
        self,
        claim: LincheckClaim,
        witness: LincheckWitness,
        transcript: Transcript,
    ) -> ProveResult[ColumnEvaluationClaim, InnerProof]:
        matrix = claim.instance.combined_row_mle(claim.row.point, claim.batch.challenge)
        state = fnp.stack([matrix, witness.assignment])
        reduced = self.sumcheck.prove(
            SumClaim(claim.batch.joint, claim.instance.s_y),
            SumcheckWitness(state),
            transcript,
        )
        return ProveResult(
            ColumnEvaluationClaim(
                reduced.reduced_claim.point, reduced.reduced_claim.value
            ),
            InnerProof(reduced.reduction_proof),
            reduced.transcript,
        )


class InnerVerifier(VerifierStage[LincheckClaim, ColumnEvaluationClaim, InnerProof]):
    """Verify lincheck conditional on a column-evaluation claim."""

    def __init__(
        self,
        *,
        sumcheck: VerifierStage[SumClaim, EvaluationClaim, Any] | None = None,
        challenges: ChallengePolicy,
    ) -> None:
        self.sumcheck = sumcheck or SumcheckVerifier(SumcheckRound(2, challenges))

    def verify(
        self,
        claim: LincheckClaim,
        reduction_proof: InnerProof,
        transcript: Transcript,
    ) -> VerifyResult[ColumnEvaluationClaim]:
        reduced = self.sumcheck.verify(
            SumClaim(claim.batch.joint, claim.instance.s_y),
            reduction_proof.sumcheck,
            transcript,
        )
        return VerifyResult(
            ColumnEvaluationClaim(
                reduced.reduced_claim.point, reduced.reduced_claim.value
            ),
            reduced.transcript,
            reduced.ok,
        )
