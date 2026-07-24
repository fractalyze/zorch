# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Spartan claim batching and the paired inner lincheck stage."""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array

from zorch.prove import fold_rounds
from zorch.spartan.engine import StageSumcheck, lincheck_engine
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import OuterOutput
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.transcript import Transcript
from zorch.verify import verify


@dataclass(frozen=True)
class BatchedClaims:
    """The RLC challenge and joint claim derived between the two stages."""

    challenge: Array
    joint: Array


def _joint_claim(claims: Array, challenge: Array) -> Array:
    """`Az + r·Bz + r²·Cz`."""
    va, vb, vc = claims[0], claims[1], claims[2]
    return va + challenge * vb + challenge * challenge * vc


def batch_claims(
    claims: Array, transcript: Transcript
) -> tuple[BatchedClaims, Transcript]:
    """Sample the batching challenge and derive the joint claim.

    This named transcript operation emits no proof. The prover and verifier
    call the same function.
    """
    transcript, sampled = transcript.sample(1)
    challenge = sampled[0]
    return BatchedClaims(challenge, _joint_claim(claims, challenge)), transcript


@dataclass(frozen=True)
class LincheckWitness:
    """Private and derived values consumed by the lincheck prover."""

    instance: R1CS
    assignment: Array
    outer: OuterOutput
    batch: BatchedClaims


@dataclass(frozen=True)
class InnerOutput:
    """The reduced lincheck claim consumed by the witness-opening stage."""

    point: Array
    final_claim: Array


@dataclass(frozen=True)
class InnerProof:
    """The inner sumcheck's round messages."""

    round_polys: Array


class InnerStage(
    Stage[LincheckWitness, InnerOutput, BatchedClaims, InnerOutput, InnerProof]
):
    """The paired prover/verifier lincheck phase."""

    name = "inner"

    def __init__(self, *, sumcheck: StageSumcheck | None = None) -> None:
        self.sumcheck = sumcheck or lincheck_engine()

    def prove(
        self, inputs: LincheckWitness, transcript: Transcript
    ) -> ProveResult[InnerOutput, InnerProof]:
        matrix = inputs.instance.combined_row_mle(
            inputs.outer.point, inputs.batch.challenge
        )
        state = fnp.stack([matrix, inputs.assignment])
        pre = transcript
        _, transcript, msgs = fold_rounds(
            self.sumcheck.prover_round, state, pre, inputs.instance.s_y
        )
        round_polys = fnp.stack(msgs)
        point, final_claim, _, _ = verify(
            self.sumcheck.verifier_round,
            inputs.batch.joint,
            round_polys,
            pre,
        )
        return ProveResult(
            InnerOutput(point, final_claim), InnerProof(round_polys), transcript
        )

    def verify(
        self,
        batch: BatchedClaims,
        proof: InnerProof,
        transcript: Transcript,
    ) -> VerifyResult[InnerOutput]:
        point, final_claim, transcript, ok = verify(
            self.sumcheck.verifier_round,
            batch.joint,
            proof.round_polys,
            transcript,
        )
        return VerifyResult(InnerOutput(point, final_claim), transcript, ok)
