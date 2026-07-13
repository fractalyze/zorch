# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Outer sumcheck (zerocheck) stage: `0 = Σ_x eq(τ,x)·(Az(x)·Bz(x) − Cz(x))`.

Samples `τ`, runs the degree-3 sumcheck over `(E=eq(τ,·), Az, Bz, Cz)`, and hands
`r_x` plus the claimed evals `(Az, Bz, Cz)(r_x)` down the chain. The per-variable
sumcheck is the shared `zorch.sumcheck` machinery; this stage adds only the `τ`
sample and the terminal identity check, so the sumcheck stays scheme-agnostic.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.spartan.carry import SpartanCarry
from zorch.spartan.r1cs import eval_eq
from zorch.spartan.summand import ZerocheckSummand
from zorch.sumcheck.prover import StandardRound
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.verify import verify

_OUTER_DEGREE = 3


def _collect_point(transcript: Transcript, round_polys: Array) -> Array:
    """Recover the sumcheck's bound point: `StandardRound` folds at each challenge
    without surfacing it, so replay the same observe→sample over a snapshot of the
    pre-rounds transcript (identical sponge + messages ⇒ identical challenges)."""
    challenges = []
    for msg in round_polys:
        transcript, r = transcript.observe_and_sample(msg, 1)
        challenges.append(r[0])
    return jnp.stack(challenges)


class OuterProver(Round):
    """Prover for the zerocheck stage; holds the matvecs `Az, Bz, Cz` as
    stage-local witness."""

    def __init__(self, az: Array, bz: Array, cz: Array) -> None:
        self.az = az
        self.bz = bz
        self.cz = cz
        self.s_x = log2_strict_usize(az.shape[0])

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, tuple[Array, Array]]:
        transcript, tau = transcript.sample(self.s_x)
        pre = transcript
        one = jnp.ones((), self.az.dtype)
        e = expand_eq_to_hypercube(tau, one)
        state = jnp.stack([e, self.az, self.bz, self.cz])
        rnd = StandardRound(ZerocheckSummand())
        final, transcript, msgs = fold_rounds(rnd, state, pre, self.s_x)
        round_polys = jnp.stack(msgs)
        r_x = _collect_point(pre, round_polys)
        claims = final[1:, 0]  # (Az, Bz, Cz)(r_x)
        transcript = transcript.observe(claims)
        carry = replace(carry, r_x=r_x, claims_outer=claims)
        return carry, transcript, (round_polys, claims)


class OuterVerifier(Round):
    """Verifier for the zerocheck stage: replay the degree-3 sumcheck, then check
    the terminal identity `eq(τ,r_x)·(vA·vB − vC) == reduced_claim`."""

    def __call__(
        self,
        carry: SpartanCarry,
        msg: tuple[Array, Array],
        transcript: Transcript,
    ) -> tuple[SpartanCarry, Transcript, Array]:
        round_polys, claims = msg
        s_x = round_polys.shape[0]
        transcript, tau = transcript.sample(s_x)
        zero = jnp.zeros((), claims.dtype)
        r_x, final_claim, transcript, ok = verify(
            SumcheckRound(_OUTER_DEGREE), zero, round_polys, transcript
        )
        transcript = transcript.observe(claims)
        va, vb, vc = claims[0], claims[1], claims[2]
        expected = eval_eq(tau, r_x) * (va * vb - vc)
        ok = ok & (final_claim == expected)
        carry = replace(carry, r_x=r_x, claims_outer=claims)
        return carry, transcript, ok
