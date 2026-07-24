# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Equality-factored sumcheck as a conditional claim reduction."""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.prove import fold_rounds
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.sumcheck.domain import natural_domain
from zorch.sumcheck.eq.eq_poly import EqPolyRound, EqPolyState
from zorch.sumcheck.prover import SumcheckSummand
from zorch.sumcheck.stage import EvaluationClaim
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript
from zorch.verify import verify


@dataclass(frozen=True)
class EqSumClaim:
    """Public sum claim for ``eq(equality_point, x) * summand(x)``."""

    equality_point: Array
    value: Array
    rounds: int


@dataclass(frozen=True)
class EqPolyWitness:
    """Factor tables witnessing an ``EqSumClaim``."""

    factors: Array


class EqPolyStage(Stage[EqSumClaim, EqPolyWitness, EvaluationClaim, Array]):
    """Prove an equality-weighted sum claim conditional on an evaluation.

    The equality polynomial remains factored instead of joining the full
    hypercube state. Factor tables therefore stay in their original field until
    the first challenge folds them.
    """

    def __init__(
        self,
        summand: SumcheckSummand,
        *,
        challenges: ChallengePolicy | None = None,
    ) -> None:
        self.summand = summand
        self.challenges = challenges or ChallengePolicy()
        self.degree = summand.degree + 1
        self.verifier_round = SumcheckRound(self.degree, self.challenges)

    @staticmethod
    def _check_claim(claim: EqSumClaim) -> None:
        if claim.equality_point.shape[0] != claim.rounds:
            raise ValueError(
                f"equality point needs {claim.rounds} coordinates, "
                f"got {claim.equality_point.shape[0]}"
            )

    def prove(
        self,
        claim: EqSumClaim,
        witness: EqPolyWitness,
        transcript: Transcript,
    ) -> ProveResult[EvaluationClaim, Array]:
        self._check_claim(claim)
        pre = transcript
        # A finite domain is valid for mixed-degree summands. Its nodes use the
        # factor field so round-0 factor evaluation stays in that field.
        domain = natural_domain(self.degree, witness.factors.dtype)
        prover_round = EqPolyRound(
            self.summand,
            claim.equality_point,
            domain,
            challenges=self.challenges,
        )
        state: EqPolyState = (
            witness.factors,
            fnp.ones(1, dtype=witness.factors.dtype),
        )
        _, _, messages = fold_rounds(prover_round, state, transcript, claim.rounds)
        reduction_proof = fnp.stack(messages)
        point, value, replayed, _ = verify(
            self.verifier_round, claim.value, reduction_proof, pre
        )
        return ProveResult(EvaluationClaim(point, value), reduction_proof, replayed)

    def verify(
        self,
        claim: EqSumClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[EvaluationClaim]:
        self._check_claim(claim)
        if reduction_proof.shape[0] != claim.rounds:
            raise ValueError(
                f"expected {claim.rounds} sumcheck rounds, "
                f"got {reduction_proof.shape[0]}"
            )
        point, value, transcript, ok = verify(
            self.verifier_round, claim.value, reduction_proof, transcript
        )
        return VerifyResult(EvaluationClaim(point, value), transcript, ok)
