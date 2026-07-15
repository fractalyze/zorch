# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""RLC combinator + inner sumcheck (lincheck) stages.

- **`Rlc*`** — samples one batching challenge `r` and folds the three outer claims
  into `joint = Az + r·Bz + r²·Cz`
  (`docs/composition/stage-composition.md`'s "RLC openings into claims under a
  fresh challenge"). Transcript-only glue, so it emits `None`.

- **`Inner*`** — a degree-2 product sumcheck proving `joint = Σ_y M(y)·z̃(y)`, with
  `M(y) = Σ_i eq(r_x)_i·(A+rB+r²C)_{i,y}` the row-bound batched matrix. It reduces
  to `(r_y, inner_final)`; the terminal `inner_final == eval_ABC · z̃(r_y)` closes
  in the PCS glue. The per-variable engine is injected (`StageSumcheck`, default
  `lincheck_engine`), so a caller can swap the algorithm / domain / wire form.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as jnp
from frx import Array

from zorch.prove import fold_rounds
from zorch.round import Bridge, Stage
from zorch.spartan.carry import SpartanCarry, _require
from zorch.spartan.engine import StageSumcheck, lincheck_engine
from zorch.spartan.r1cs import R1CS
from zorch.transcript import Transcript
from zorch.verify import verify


def _joint_claim(claims_outer: Array, r: Array) -> Array:
    """`Az + r·Bz + r²·Cz` — the three outer claims batched by powers of `r`."""
    va, vb, vc = claims_outer[0], claims_outer[1], claims_outer[2]
    return va + r * vb + r * r * vc


class RlcProver(Bridge):
    """Sample the batching challenge; fold the outer claims into `joint_claim`."""

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, None]:
        claims = _require(carry.claims_outer, "claims_outer", "outer")
        transcript, r = transcript.sample(1)
        r = r[0]
        carry = replace(carry, r_batch=r, joint_claim=_joint_claim(claims, r))
        return carry, transcript, None


class RlcVerifier(Bridge):
    """Verifier dual of `RlcProver`: replay the same sample and fold."""

    def __call__(
        self, carry: SpartanCarry, msg: None, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        del msg  # transcript-only round — the prover sends no proof message
        claims = _require(carry.claims_outer, "claims_outer", "outer")
        transcript, r = transcript.sample(1)
        r = r[0]
        carry = replace(carry, r_batch=r, joint_claim=_joint_claim(claims, r))
        return carry, transcript, jnp.bool_(True)


class InnerProver(Stage):
    """Prover for the lincheck stage; holds the instance and assignment `z` as
    stage-local witness. Inject `sumcheck` to swap the per-variable engine."""

    def __init__(
        self, instance: R1CS, z: Array, *, sumcheck: StageSumcheck | None = None
    ) -> None:
        self.instance = instance
        self.z = z
        self.sumcheck = sumcheck or lincheck_engine()

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        r_x = _require(carry.r_x, "r_x", "outer")
        r = _require(carry.r_batch, "r_batch", "RLC")
        joint = _require(carry.joint_claim, "joint_claim", "RLC")
        m = self.instance.combined_row_mle(r_x, r)
        state = jnp.stack([m, self.z])
        pre = transcript
        _, transcript, msgs = fold_rounds(
            self.sumcheck.prover_round, state, pre, self.instance.s_y
        )
        round_polys = jnp.stack(msgs)
        # Recover r_y by replaying the injected verifier round (point is
        # independent of the claim value).
        r_y, _, _, _ = verify(self.sumcheck.verifier_round, joint, round_polys, pre)
        carry = replace(carry, r_y=r_y)
        return carry, transcript, round_polys


class InnerVerifier(Stage):
    """Verifier for the lincheck sumcheck: reduce `joint_claim` to `(r_y,
    inner_final)`. The terminal identity closes in the PCS glue."""

    def __init__(self, *, sumcheck: StageSumcheck | None = None) -> None:
        self.sumcheck = sumcheck or lincheck_engine()

    def __call__(
        self, carry: SpartanCarry, msg: Array, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        joint = _require(carry.joint_claim, "joint_claim", "RLC")
        r_y, inner_final, transcript, ok = verify(
            self.sumcheck.verifier_round, joint, msg, transcript
        )
        carry = replace(carry, r_y=r_y, inner_final=inner_final)
        return carry, transcript, ok
