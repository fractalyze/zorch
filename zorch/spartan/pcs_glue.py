# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The paired witness-opening stage and transparent test PCS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.poly.multilinear import eval_mle
from zorch.spartan.lincheck import BatchedClaims, InnerOutput
from zorch.spartan.r1cs import R1CS, eval_public_half, recombine_z_eval
from zorch.spartan.zerocheck import OuterOutput
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class WitnessOpenData:
    """Private PCS state and evaluation point used to open the witness."""

    pcs: PcsProver[Any, Any, Any]
    prover_data: Any
    point: Array


@dataclass(frozen=True)
class WitnessOpeningClaim:
    """The terminal identity closed by opening the committed witness."""

    point: Array
    final_claim: Array
    matrix_eval: Array
    public_eval: Array


@dataclass(frozen=True)
class WitnessOpeningStatement:
    """Public commitment and terminal claim checked by the opening verifier."""

    pcs: PcsVerifier[Any, Any]
    commitment: Array
    claim: WitnessOpeningClaim


@dataclass(frozen=True)
class WitnessOpenProof:
    values: Array
    pcs_proof: Any


def witness_opening_claim(
    instance: R1CS,
    public_inputs: Array,
    outer: OuterOutput,
    batch: BatchedClaims,
    inner: InnerOutput,
) -> WitnessOpeningClaim:
    """Derive the terminal opening claim from the pipeline's typed outputs."""
    public_eval = eval_public_half(
        public_inputs, inner.point[1:], instance.num_vars_padded
    )
    matrix_eval = instance.eval_combined_matrix(
        outer.point, inner.point, batch.challenge
    )
    return WitnessOpeningClaim(inner.point, inner.final_claim, matrix_eval, public_eval)


class WitnessOpenStage(
    Stage[
        WitnessOpenData,
        None,
        WitnessOpeningStatement,
        None,
        WitnessOpenProof,
    ]
):
    """Open the committed witness and close the lincheck's terminal identity."""

    name = "witness_open"

    def prove(
        self, inputs: WitnessOpenData, transcript: Transcript
    ) -> ProveResult[None, WitnessOpenProof]:
        values, proof, transcript = inputs.pcs.open(
            inputs.prover_data, [inputs.point[1:]], transcript
        )
        return ProveResult(None, WitnessOpenProof(values, proof), transcript)

    def verify(
        self,
        inputs: WitnessOpeningStatement,
        proof: WitnessOpenProof,
        transcript: Transcript,
    ) -> VerifyResult[None]:
        claim = inputs.claim
        ok_open, transcript = inputs.pcs.verify(
            inputs.commitment,
            [claim.point[1:]],
            proof.values,
            proof.pcs_proof,
            transcript,
        )
        eval_w = proof.values[0]
        z_eval = recombine_z_eval(eval_w, claim.public_eval, claim.point[0])
        ok_final = claim.final_claim == claim.matrix_eval * z_eval
        return VerifyResult(None, transcript, ok_open & ok_final)


class DensePcs:
    """A transparent multilinear "PCS" for tests: the commitment IS the evals."""

    def commit(self, polys: Sequence[Array]) -> tuple[Array, tuple[Array, ...]]:
        return polys[0], tuple(polys)

    def open(
        self,
        prover_data: tuple[Array, ...],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, None, Transcript]:
        poly = prover_data[0]
        values = fnp.stack([eval_mle(poly, pt) for pt in points])
        return values, None, transcript

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: None,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        del proof
        recomputed = fnp.stack([eval_mle(commitment, pt) for pt in points])
        return fnp.all(recomputed == values), transcript
