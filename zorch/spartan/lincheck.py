# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Spartan claim batching and the inner lincheck reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import RowEvaluationClaim
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.sumcheck.prover import ProductSummand, StandardRound
from zorch.sumcheck.stage import (
    EvaluationClaim,
    SumcheckStage,
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
    """``Az + r·Bz + r²·Cz``."""
    va, vb, vc = claims[0], claims[1], claims[2]
    return va + challenge * vb + challenge * challenge * vc


def batch_claims(
    claims: Array,
    transcript: Transcript,
    challenges: ChallengePolicy | None = None,
) -> tuple[BatchedClaims, Transcript]:
    """Sample the batching challenge and derive the joint value.

    The claims must already have been absorbed into ``transcript``. This named
    protocol operation emits no proof; both roles call it at the same point.
    """
    policy = challenges or ChallengePolicy()
    transcript, challenge = policy.sample(transcript)
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


class InnerStage(
    Stage[LincheckClaim, LincheckWitness, ColumnEvaluationClaim, InnerProof]
):
    """Prove lincheck conditional on a column-evaluation claim."""

    def __init__(
        self,
        *,
        sumcheck: Stage[SumClaim, SumcheckWitness, EvaluationClaim, Any] | None = None,
        challenges: ChallengePolicy | None = None,
    ) -> None:
        self.challenges = challenges or ChallengePolicy()
        self.sumcheck = sumcheck or SumcheckStage(
            StandardRound(ProductSummand(2), challenges=self.challenges),
            SumcheckRound(2, self.challenges),
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
        reduced_claim = ColumnEvaluationClaim(
            reduced.reduced_claim.point, reduced.reduced_claim.value
        )
        return ProveResult(
            reduced_claim,
            InnerProof(reduced.reduction_proof),
            reduced.transcript,
        )

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
        reduced_claim = ColumnEvaluationClaim(
            reduced.reduced_claim.point, reduced.reduced_claim.value
        )
        return VerifyResult(reduced_claim, reduced.transcript, reduced.ok)
