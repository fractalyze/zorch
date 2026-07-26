# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Transparent multilinear PCS used only by Spartan tests."""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
from frx import Array

from zorch.pcs.stage import OpeningClaim, OpeningProof, OpeningWitness
from zorch.poly.multilinear import eval_mle
from zorch.stage import (
    ProveResult,
    TrivialClaim,
    VerifyResult,
)
from zorch.transcript import Transcript


class DensePcs:
    """A transparent test PCS whose commitment is the polynomial evaluation table."""

    def commit(self, polys: Sequence[Array]) -> tuple[Array, tuple[Array, ...]]:
        return polys[0], tuple(polys)

    def prove(
        self,
        claim: OpeningClaim[Array],
        witness: OpeningWitness[tuple[Array, ...]],
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof[Array]]:
        values, proof, transcript = self._open(
            witness.prover_data, claim.points, transcript
        )
        return ProveResult(TrivialClaim(), OpeningProof(values, proof), transcript)

    def _open(
        self,
        prover_data: tuple[Array, ...],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, None, Transcript]:
        poly = prover_data[0]
        values = fnp.stack([eval_mle(poly, point) for point in points])
        return values, None, transcript

    def verify(
        self,
        claim: OpeningClaim[Array],
        reduction_proof: OpeningProof[Array],
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        ok, transcript = self._verify_opening(
            claim.commitment,
            claim.points,
            reduction_proof.values,
            reduction_proof.proof,
            transcript,
        )
        return VerifyResult(TrivialClaim(), transcript, ok)

    def _verify_opening(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: None,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        del proof
        recomputed = fnp.stack([eval_mle(commitment, point) for point in points])
        return fnp.all(recomputed == values), transcript
