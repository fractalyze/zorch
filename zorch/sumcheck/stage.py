# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck roles reducing a sum claim to an evaluation claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
from frx import Array

from zorch.prove import fold_rounds
from zorch.round import ProverRound, RunningClaim, VerifierRound
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.sumcheck.prover import initial_carry
from zorch.transcript import Transcript
from zorch.verify import verify


@dataclass(frozen=True)
class SumClaim:
    """Public claim that a polynomial sums to ``value`` over a Boolean cube."""

    value: Array
    rounds: int


@dataclass(frozen=True)
class SumcheckWitness:
    """Dense factor tables witnessing a ``SumClaim``."""

    state: Array


@dataclass(frozen=True)
class EvaluationClaim:
    """Claim that the reduced polynomial evaluates to ``value`` at ``point``."""

    point: Array
    value: Array


class SumcheckProver(ProverStage[SumClaim, SumcheckWitness, EvaluationClaim, Array]):
    """The prover role of dense sumcheck."""

    def __init__(
        self,
        prover_round: ProverRound[Any, Array],
        verifier_round: VerifierRound[RunningClaim, Array],
    ) -> None:
        self.prover_round = prover_round
        # Kept for the pairing, not for proving: the prover reduces its own claim.
        self.verifier_round = verifier_round

    def prove(
        self,
        claim: SumClaim,
        witness: SumcheckWitness,
        transcript: Transcript,
    ) -> ProveResult[EvaluationClaim, Array]:
        (_, reduced), transcript, messages = fold_rounds(
            self.prover_round,
            initial_carry(witness.state, claim.value, claim.rounds),
            transcript,
            claim.rounds,
        )
        return ProveResult(
            EvaluationClaim(reduced.point, reduced.value),
            fnp.stack(messages),
            transcript,
        )


class SumcheckVerifier(VerifierStage[SumClaim, EvaluationClaim, Array]):
    """The verifier role of dense sumcheck."""

    def __init__(self, verifier_round: VerifierRound[RunningClaim, Array]) -> None:
        self.verifier_round = verifier_round

    def verify(
        self,
        claim: SumClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[EvaluationClaim]:
        if reduction_proof.shape[0] != claim.rounds:
            raise ValueError(
                f"expected {claim.rounds} sumcheck rounds, "
                f"got {reduction_proof.shape[0]}"
            )
        point, value, transcript, ok = verify(
            self.verifier_round, claim.value, reduction_proof, transcript
        )
        return VerifyResult(EvaluationClaim(point, value), transcript, ok)


if TYPE_CHECKING:
    _p: type[ProverStage[(SumClaim, SumcheckWitness, EvaluationClaim, Array)]] = (
        SumcheckProver
    )
    _v: type[VerifierStage[(SumClaim, EvaluationClaim, Array)]] = SumcheckVerifier
