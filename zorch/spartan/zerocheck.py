# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Outer sumcheck (zerocheck) stage: `0 = Σ_x eq(τ,x)·(Az(x)·Bz(x) − Cz(x))`.

Samples `τ`, runs a degree-3 sumcheck over `(E=eq(τ,·), Az, Bz, Cz)`, and hands
`r_x` plus the claimed evals `(Az, Bz, Cz)(r_x)` down the chain. The per-variable
sumcheck engine is injected (`StageSumcheck`, default `zerocheck_engine`), so a
caller can swap the algorithm / domain / wire form; this stage adds only the `τ`
sample and the terminal identity check.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
from frx import Array

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.prove import fold_rounds
from zorch.round import Stage
from zorch.spartan.carry import SpartanCarry
from zorch.spartan.engine import StageSumcheck, zerocheck_engine
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.verify import verify


class OuterProver(Stage):
    """Prover for the zerocheck stage; holds the matvecs `Az, Bz, Cz` as
    stage-local witness. Inject `sumcheck` to swap the per-variable engine."""

    def __init__(
        self,
        az: Array,
        bz: Array,
        cz: Array,
        *,
        sumcheck: StageSumcheck | None = None,
    ) -> None:
        self.az = az
        self.bz = bz
        self.cz = cz
        self.s_x = log2_strict_usize(az.shape[0])
        self.sumcheck = sumcheck or zerocheck_engine()

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, tuple[Array, Array]]:
        transcript, tau = transcript.sample(self.s_x)
        pre = transcript
        one = fnp.ones((), self.az.dtype)
        e = expand_eq_to_hypercube(tau, one)
        state = fnp.stack([e, self.az, self.bz, self.cz])
        _, transcript, msgs = fold_rounds(
            self.sumcheck.prover_round, state, pre, self.s_x
        )
        round_polys = fnp.stack(msgs)
        # Recover r_x by replaying the injected verifier round — wire-agnostic, so
        # the claim value (0) does not affect the sampled point.
        zero = fnp.zeros((), self.az.dtype)
        r_x, _, _, _ = verify(self.sumcheck.verifier_round, zero, round_polys, pre)
        # Claimed evals straight from the MLEs — independent of the engine's fold
        # representation.
        claims = fnp.stack(
            [eval_mle(self.az, r_x), eval_mle(self.bz, r_x), eval_mle(self.cz, r_x)]
        )
        transcript = transcript.observe(claims)
        carry = replace(carry, r_x=r_x, claims_outer=claims)
        return carry, transcript, (round_polys, claims)


class OuterVerifier(Stage):
    """Verifier for the zerocheck stage: replay the sumcheck, then check the
    terminal identity `eq(τ,r_x)·(vA·vB − vC) == reduced_claim`."""

    def __init__(self, *, sumcheck: StageSumcheck | None = None) -> None:
        self.sumcheck = sumcheck or zerocheck_engine()

    def __call__(
        self,
        carry: SpartanCarry,
        msg: tuple[Array, Array],
        transcript: Transcript,
    ) -> tuple[SpartanCarry, Transcript, Array]:
        round_polys, claims = msg
        s_x = round_polys.shape[0]
        transcript, tau = transcript.sample(s_x)
        zero = fnp.zeros((), claims.dtype)
        r_x, final_claim, transcript, ok = verify(
            self.sumcheck.verifier_round, zero, round_polys, transcript
        )
        transcript = transcript.observe(claims)
        va, vb, vc = claims[0], claims[1], claims[2]
        expected = eval_eq(tau, r_x) * (va * vb - vc)
        ok = ok & (final_claim == expected)
        carry = replace(carry, r_x=r_x, claims_outer=claims)
        return carry, transcript, ok
