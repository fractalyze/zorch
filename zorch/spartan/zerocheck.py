# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Outer Spartan zerocheck as one paired, typed stage."""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.prove import fold_rounds
from zorch.spartan.engine import StageSumcheck, zerocheck_engine
from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.verify import verify


@dataclass(frozen=True)
class OuterPolynomials:
    """The three multilinear polynomials constrained by zerocheck."""

    az: Array
    bz: Array
    cz: Array


@dataclass(frozen=True)
class OuterOutput:
    """Claims produced for the later lincheck and witness-opening stages."""

    point: Array
    claims: Array


@dataclass(frozen=True)
class OuterProof:
    """The outer sumcheck messages and terminal `(Az, Bz, Cz)` evaluations."""

    round_polys: Array
    claims: Array


class OuterStage(Stage[OuterPolynomials, OuterOutput, None, OuterOutput, OuterProof]):
    """The paired prover/verifier zerocheck phase."""

    name = "outer"

    def __init__(self, *, sumcheck: StageSumcheck | None = None) -> None:
        self.sumcheck = sumcheck or zerocheck_engine()

    def prove(
        self, inputs: OuterPolynomials, transcript: Transcript
    ) -> ProveResult[OuterOutput, OuterProof]:
        az, bz, cz = inputs.az, inputs.bz, inputs.cz
        s_x = log2_strict_usize(az.shape[0])
        transcript, tau = transcript.sample(s_x)
        pre = transcript
        one = fnp.ones((), az.dtype)
        state = fnp.stack([expand_eq_to_hypercube(tau, one), az, bz, cz])
        _, transcript, msgs = fold_rounds(self.sumcheck.prover_round, state, pre, s_x)
        round_polys = fnp.stack(msgs)
        zero = fnp.zeros((), az.dtype)
        point, _, _, _ = verify(self.sumcheck.verifier_round, zero, round_polys, pre)
        claims = fnp.stack(
            [eval_mle(az, point), eval_mle(bz, point), eval_mle(cz, point)]
        )
        transcript = transcript.observe(claims)
        output = OuterOutput(point, claims)
        return ProveResult(output, OuterProof(round_polys, claims), transcript)

    def verify(
        self,
        inputs: None,
        proof: OuterProof,
        transcript: Transcript,
    ) -> VerifyResult[OuterOutput]:
        del inputs
        s_x = proof.round_polys.shape[0]
        transcript, tau = transcript.sample(s_x)
        zero = fnp.zeros((), proof.claims.dtype)
        point, final_claim, transcript, ok = verify(
            self.sumcheck.verifier_round, zero, proof.round_polys, transcript
        )
        transcript = transcript.observe(proof.claims)
        va, vb, vc = proof.claims[0], proof.claims[1], proof.claims[2]
        ok = ok & (final_claim == eval_eq(tau, point) * (va * vb - vc))
        return VerifyResult(OuterOutput(point, proof.claims), transcript, ok)
