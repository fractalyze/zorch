# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable roles for the terminal witness-opening reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frx import Array

from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.spartan.lincheck import BatchedClaims, ColumnEvaluationClaim
from zorch.spartan.r1cs import R1CS, eval_public_half, recombine_z_eval
from zorch.spartan.zerocheck import RowEvaluationClaim
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class WitnessOpeningClaim:
    """Public terminal claim closed by opening the committed witness."""

    commitment: Array
    point: Array
    product_value: Array
    matrix_value: Array
    public_value: Array


@dataclass(frozen=True)
class WitnessOpeningWitness:
    """Private PCS opening state for ``WitnessOpeningClaim``."""

    prover_data: Any


@dataclass(frozen=True)
class WitnessOpenProof:
    """Witness evaluation and the underlying PCS opening proof."""

    values: Array
    pcs_proof: Any


def witness_opening_claim(
    commitment: Array,
    instance: R1CS,
    public_inputs: Array,
    row: RowEvaluationClaim,
    batch: BatchedClaims,
    column: ColumnEvaluationClaim,
) -> WitnessOpeningClaim:
    """Derive the terminal public claim from preceding reduced claims."""
    public_value = eval_public_half(
        public_inputs, column.point[1:], instance.num_vars_padded
    )
    matrix_value = instance.eval_combined_matrix(
        row.point, column.point, batch.challenge
    )
    return WitnessOpeningClaim(
        commitment,
        column.point,
        column.value,
        matrix_value,
        public_value,
    )


class WitnessOpenProver(
    ProverStage[WitnessOpeningClaim, WitnessOpeningWitness, None, WitnessOpenProof]
):
    """Prove a committed-witness opening; owns only the PCS prover capability."""

    def __init__(self, pcs_prover: PcsProver[Any, Any, Any]) -> None:
        self.pcs_prover = pcs_prover

    def prove(
        self,
        claim: WitnessOpeningClaim,
        witness: WitnessOpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[None, WitnessOpenProof]:
        values, pcs_proof, transcript = self.pcs_prover.open(
            witness.prover_data, [claim.point[1:]], transcript
        )
        return ProveResult(None, WitnessOpenProof(values, pcs_proof), transcript)


class WitnessOpenVerifier(VerifierStage[WitnessOpeningClaim, None, WitnessOpenProof]):
    """Verify a committed-witness opening; owns only the PCS verifier capability."""

    def __init__(self, pcs_verifier: PcsVerifier[Any, Any]) -> None:
        self.pcs_verifier = pcs_verifier

    def verify(
        self,
        claim: WitnessOpeningClaim,
        reduction_proof: WitnessOpenProof,
        transcript: Transcript,
    ) -> VerifyResult[None]:
        ok_open, transcript = self.pcs_verifier.verify(
            claim.commitment,
            [claim.point[1:]],
            reduction_proof.values,
            reduction_proof.pcs_proof,
            transcript,
        )
        eval_w = reduction_proof.values[0]
        z_eval = recombine_z_eval(eval_w, claim.public_value, claim.point[0])
        ok_final = claim.product_value == claim.matrix_value * z_eval
        return VerifyResult(None, transcript, ok_open & ok_final)
