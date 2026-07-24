# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck as a conditional reduction from a sum claim to an evaluation claim."""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array

from zorch.prove import fold_rounds
from zorch.round import InnerVerifierRound, ProverRound
from zorch.stage import ProveResult, Stage, VerifyResult
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


class SumcheckStage(Stage[SumClaim, SumcheckWitness, EvaluationClaim, Array]):
    """Prove a sum claim conditional on the returned evaluation claim.

    The configured rounds are recurrence kernels. This stage owns their
    scheduling, proof assembly, transcript advancement, and verification.
    """

    def __init__(
        self,
        prover_round: ProverRound,
        verifier_round: InnerVerifierRound,
    ) -> None:
        self.prover_round = prover_round
        self.verifier_round = verifier_round

    def prove(
        self,
        claim: SumClaim,
        witness: SumcheckWitness,
        transcript: Transcript,
    ) -> ProveResult[EvaluationClaim, Array]:
        pre = transcript
        _, _, messages = fold_rounds(
            self.prover_round, witness.state, transcript, claim.rounds
        )
        reduction_proof = fnp.stack(messages)
        # Verifier replay is authoritative for both the reduced claim and the
        # transcript. An invalid witness still produces a proof that verification
        # can reject against the independently supplied source claim.
        point, value, replayed, _ = verify(
            self.verifier_round, claim.value, reduction_proof, pre
        )
        return ProveResult(EvaluationClaim(point, value), reduction_proof, replayed)

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
