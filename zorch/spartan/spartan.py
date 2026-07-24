# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Spartan as a composite stage with explicit DAG-shaped orchestration.

Outer claims feed both the inner phase and witness opening. The commitment and
PCS prover data flow directly to the witness-opening phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frx import Array

from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.spartan.engine import StageSumcheck
from zorch.spartan.lincheck import (
    InnerProof,
    InnerStage,
    LincheckWitness,
    batch_claims,
)
from zorch.spartan.pcs_glue import (
    WitnessOpenData,
    WitnessOpeningStatement,
    WitnessOpenProof,
    WitnessOpenStage,
    witness_opening_claim,
)
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import (
    OuterPolynomials,
    OuterProof,
    OuterStage,
)
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class SpartanWitness:
    """Private assignment, public statement, and PCS prover for Spartan."""

    instance: R1CS
    assignment: Array
    public_inputs: Array
    pcs: PcsProver[Any, Any, Any]


@dataclass(frozen=True)
class SpartanStatement:
    """Public R1CS statement and PCS verifier for Spartan."""

    instance: R1CS
    public_inputs: Array
    pcs: PcsVerifier[Any, Any]


@dataclass(frozen=True)
class SpartanProof:
    """One named proof section per coarse protocol phase."""

    commitment: Array
    outer: OuterProof
    inner: InnerProof
    witness_open: WitnessOpenProof


class Spartan(Stage[SpartanWitness, None, SpartanStatement, None, SpartanProof]):
    """The composite Spartan protocol stage."""

    name = "spartan"

    def __init__(
        self,
        *,
        outer: StageSumcheck | None = None,
        inner: StageSumcheck | None = None,
    ) -> None:
        self.outer = OuterStage(sumcheck=outer)
        self.inner = InnerStage(sumcheck=inner)
        self.witness_open = WitnessOpenStage()

    @staticmethod
    def _absorb_statement(
        transcript: Transcript, commitment: Array, public_inputs: Array
    ) -> Transcript:
        transcript = transcript.observe(commitment)
        if public_inputs.shape[0] > 0:
            transcript = transcript.observe(public_inputs)
        return transcript

    def prove(
        self, inputs: SpartanWitness, transcript: Transcript
    ) -> ProveResult[None, SpartanProof]:
        instance = inputs.instance
        assignment = inputs.assignment
        witness = assignment[: instance.num_vars_padded]
        commitment, prover_data = inputs.pcs.commit([witness])
        transcript = self._absorb_statement(
            transcript, commitment, inputs.public_inputs
        )

        az, bz, cz = instance.matvecs(assignment)
        outer = self.outer.prove(OuterPolynomials(az, bz, cz), transcript)
        batch, transcript = batch_claims(outer.output.claims, outer.transcript)
        inner = self.inner.prove(
            LincheckWitness(instance, assignment, outer.output, batch), transcript
        )
        opening = self.witness_open.prove(
            WitnessOpenData(inputs.pcs, prover_data, inner.output.point),
            inner.transcript,
        )
        proof = SpartanProof(
            commitment,
            outer.proof,
            inner.proof,
            opening.proof,
        )
        return ProveResult(None, proof, opening.transcript)

    def verify(
        self,
        inputs: SpartanStatement,
        proof: SpartanProof,
        transcript: Transcript,
    ) -> VerifyResult[None]:
        transcript = self._absorb_statement(
            transcript, proof.commitment, inputs.public_inputs
        )
        outer = self.outer.verify(None, proof.outer, transcript)
        batch, transcript = batch_claims(outer.output.claims, outer.transcript)
        inner = self.inner.verify(batch, proof.inner, transcript)
        claim = witness_opening_claim(
            inputs.instance,
            inputs.public_inputs,
            outer.output,
            batch,
            inner.output,
        )
        opening = self.witness_open.verify(
            WitnessOpeningStatement(inputs.pcs, proof.commitment, claim),
            proof.witness_open,
            inner.transcript,
        )
        ok = outer.ok & inner.ok & opening.ok
        return VerifyResult(None, opening.transcript, ok)
