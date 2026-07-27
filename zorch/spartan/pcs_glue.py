# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable roles for the terminal witness-opening reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frx import Array

from zorch.pcs.stage import (
    OpeningClaim,
    OpeningProof,
    OpeningWitness,
)
from zorch.spartan.lincheck import BatchedClaims, ColumnEvaluationClaim
from zorch.spartan.r1cs import R1CS, eval_public_half, recombine_z_eval
from zorch.spartan.zerocheck import RowEvaluationClaim
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
    VerifierStage,
    VerifyResult,
)
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
    ProverStage[
        WitnessOpeningClaim, WitnessOpeningWitness, TrivialClaim, WitnessOpenProof
    ]
):
    """Prove a committed-witness opening; owns only the PCS prover capability."""

    def __init__(
        self,
        pcs_prover: ProverStage[
            OpeningClaim[Any], OpeningWitness[Any], TrivialClaim, OpeningProof[Any]
        ],
    ) -> None:
        self.pcs_prover = pcs_prover

    def prove(
        self,
        claim: WitnessOpeningClaim,
        witness: WitnessOpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, WitnessOpenProof]:
        opened = self.pcs_prover.prove(
            OpeningClaim(claim.commitment, [claim.point[1:]]),
            OpeningWitness(witness.prover_data),
            transcript,
        )
        inner = opened.reduction_proof
        return ProveResult(
            TrivialClaim(),
            WitnessOpenProof(inner.values, inner.proof),
            opened.transcript,
        )


class WitnessOpenVerifier(
    VerifierStage[WitnessOpeningClaim, TrivialClaim, WitnessOpenProof]
):
    """Verify a committed-witness opening; owns only the PCS verifier capability."""

    def __init__(
        self,
        pcs_verifier: VerifierStage[OpeningClaim[Any], TrivialClaim, OpeningProof[Any]],
    ) -> None:
        self.pcs_verifier = pcs_verifier

    def verify(
        self,
        claim: WitnessOpeningClaim,
        reduction_proof: WitnessOpenProof,
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        verified = self.pcs_verifier.verify(
            OpeningClaim(claim.commitment, [claim.point[1:]]),
            OpeningProof(reduction_proof.values, reduction_proof.pcs_proof),
            transcript,
        )
        ok_open, transcript = verified.ok, verified.transcript
        eval_w = reduction_proof.values[0]
        z_eval = recombine_z_eval(eval_w, claim.public_value, claim.point[0])
        ok_final = claim.product_value == claim.matrix_value * z_eval
        return VerifyResult(TrivialClaim(), transcript, ok_open & ok_final)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _p: type[
        ProverStage[
            (WitnessOpeningClaim, WitnessOpeningWitness, TrivialClaim, WitnessOpenProof)
        ]
    ] = WitnessOpenProver
    _v: type[VerifierStage[(WitnessOpeningClaim, TrivialClaim, WitnessOpenProof)]] = (
        WitnessOpenVerifier
    )
