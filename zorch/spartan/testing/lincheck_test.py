# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The RLC combinator + inner (lincheck) stage, over two prime fields.

Like the outer stage these carry no field of their own, so the roundtrips and
soundness checks run over both koalabear and babybear, pinning the agnosticism.
"""
from __future__ import annotations

from typing import Any, cast

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import Array

from zorch.spartan.carry import SpartanCarry
from zorch.spartan.lincheck import (
    InnerProver,
    InnerVerifier,
    RlcProver,
    RlcVerifier,
    _joint_claim,
)
from zorch.spartan.r1cs import R1CS
from zorch.spartan.testing.toy import toy_r1cs
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont
BB = zk_dtypes.babybear_mont
FIELDS = (("koalabear", KB), ("babybear", BB))


class RlcCombinatorTest(parameterized.TestCase):
    @parameterized.named_parameters(*FIELDS)
    def test_joint_claim_is_powers_of_r(self, dtype: Any) -> None:
        claims = rand_field(1, (3,), dtype)
        r = fnp.asarray(rand_field(2, (1,), dtype)[0])
        got = _joint_claim(claims, r)
        want = claims[0] + r * claims[1] + r * r * claims[2]
        self.assertTrue(bool(got == want))

    @parameterized.named_parameters(*FIELDS)
    def test_prover_verifier_agree(self, dtype: Any) -> None:
        claims = rand_field(3, (3,), dtype)
        carry = SpartanCarry(claims_outer=claims)
        pc, _, msg = RlcProver()(carry, cheap_transcript(dtype))
        vc, _, ok = RlcVerifier()(carry, msg, cheap_transcript(dtype))
        self.assertIsNone(msg)  # glue emits no proof
        self.assertTrue(bool(ok))
        self.assertTrue(bool(pc.r_batch == vc.r_batch))
        self.assertTrue(bool(pc.joint_claim == vc.joint_claim))


class InnerStageTest(parameterized.TestCase):
    def _setup(self, seed: int, dtype: Any) -> tuple[R1CS, Array, SpartanCarry]:
        inst, z, _, _ = toy_r1cs(seed, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        # Hand the inner stage a carry as if outer + RLC already ran.
        r_x = rand_field(seed + 10, (inst.s_x,), dtype)
        r = fnp.asarray(rand_field(seed + 11, (1,), dtype)[0])
        joint = fnp.sum(inst.combined_row_mle(r_x, r) * z)
        carry = SpartanCarry(r_x=r_x, r_batch=r, joint_claim=joint)
        return inst, z, carry

    @parameterized.named_parameters(*FIELDS)
    def test_roundtrip_accepts(self, dtype: Any) -> None:
        inst, z, carry = self._setup(20, dtype)
        pc, _, msg = InnerProver(inst, z)(carry, cheap_transcript(dtype))
        vc, _, ok = InnerVerifier()(carry, msg, cheap_transcript(dtype))
        self.assertTrue(bool(ok))
        self.assertTrue(bool(fnp.all(pc.r_y == vc.r_y)))

    @parameterized.named_parameters(*FIELDS)
    def test_wrong_joint_claim_rejected(self, dtype: Any) -> None:
        inst, z, carry = self._setup(21, dtype)
        _, _, msg = InnerProver(inst, z)(carry, cheap_transcript(dtype))
        bad = SpartanCarry(
            r_x=carry.r_x,
            r_batch=carry.r_batch,
            joint_claim=cast(Array, carry.joint_claim) + fnp.ones((), dtype),
        )
        _, _, ok = InnerVerifier()(bad, msg, cheap_transcript(dtype))
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
