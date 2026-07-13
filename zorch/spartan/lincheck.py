# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""RLC combinator + inner sumcheck (lincheck) stages.

- **`Rlc*`** — samples one batching challenge `r` and folds the three outer claims
  into `joint = Az + r·Bz + r²·Cz` (`docs/stage-composition.md`'s "RLC openings
  into claims under a fresh challenge"). Transcript-only glue, so it emits `None`.

- **`Inner*`** — a degree-2 product sumcheck proving `joint = Σ_y M(y)·z̃(y)`, with
  `M(y) = Σ_i eq(r_x)_i·(A+rB+r²C)_{i,y}` the row-bound batched matrix. It reduces
  to `(r_y, inner_final)`; the terminal `inner_final == eval_ABC · z̃(r_y)` closes
  in the PCS glue, once `z̃(r_y)` comes from the witness opening.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.spartan.carry import SpartanCarry, _require
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import _collect_point
from zorch.sumcheck.prover import ProductSummand, StandardRound
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript

_INNER_DEGREE = 2


def _joint_claim(claims_outer: Array, r: Array) -> Array:
    """`Az + r·Bz + r²·Cz` — the three outer claims batched by powers of `r`."""
    va, vb, vc = claims_outer[0], claims_outer[1], claims_outer[2]
    return va + r * vb + r * r * vc


class RlcProver(Round):
    """Sample the batching challenge; fold the outer claims into `joint_claim`."""

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, None]:
        claims = _require(carry.claims_outer, "claims_outer", "outer")
        transcript, r = transcript.sample(1)
        r = r[0]
        carry = replace(carry, r_batch=r, joint_claim=_joint_claim(claims, r))
        return carry, transcript, None


class RlcVerifier(Round):
    """Verifier dual of `RlcProver`: replay the same sample and fold."""

    def __call__(
        self, carry: SpartanCarry, msg: None, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        del msg  # glue round carries no proof
        claims = _require(carry.claims_outer, "claims_outer", "outer")
        transcript, r = transcript.sample(1)
        r = r[0]
        carry = replace(carry, r_batch=r, joint_claim=_joint_claim(claims, r))
        return carry, transcript, jnp.bool_(True)


class InnerProver(Round):
    """Prover for the lincheck stage; holds the instance and assignment `z` as
    stage-local witness."""

    def __init__(self, instance: R1CS, z: Array) -> None:
        self.instance = instance
        self.z = z

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        r_x = _require(carry.r_x, "r_x", "outer")
        r = _require(carry.r_batch, "r_batch", "RLC")
        m = self.instance.combined_row_mle(r_x, r)
        state = jnp.stack([m, self.z])
        pre = transcript
        rnd = StandardRound(ProductSummand(_INNER_DEGREE))
        _, transcript, msgs = fold_rounds(rnd, state, pre, self.instance.s_y)
        round_polys = jnp.stack(msgs)
        r_y = _collect_point(pre, round_polys)
        carry = replace(carry, r_y=r_y)
        return carry, transcript, round_polys


class InnerVerifier(Round):
    """Verifier for the lincheck sumcheck: reduce `joint_claim` to `(r_y,
    inner_final)`. The terminal identity closes in the PCS glue."""

    def __call__(
        self, carry: SpartanCarry, msg: Array, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        joint = _require(carry.joint_claim, "joint_claim", "RLC")
        r_y, inner_final, transcript, ok = _verify_inner(joint, msg, transcript)
        carry = replace(carry, r_y=r_y, inner_final=inner_final)
        return carry, transcript, ok


def _verify_inner(
    joint: Array, round_polys: Array, transcript: Transcript
) -> tuple[Array, Array, Transcript, Array]:
    from zorch.verify import verify

    return verify(SumcheckRound(_INNER_DEGREE), joint, round_polys, transcript)
