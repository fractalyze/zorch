# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import cast

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.poly.multilinear import eval_mle
from zorch.spartan.carry import SpartanCarry
from zorch.spartan.pcs_glue import (
    DensePcs,
    WitnessOpenProver,
    WitnessOpenVerifier,
)
from zorch.spartan.r1cs import R1CS, eval_public_half, recombine_z_eval
from zorch.spartan.testing.toy import toy_r1cs
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


class DensePcsTest(absltest.TestCase):
    def test_open_verify_roundtrip(self) -> None:
        poly = rand_field(1, (8,), KB)
        pt = rand_field(2, (3,), KB)
        pcs = DensePcs()
        comm, pdata = pcs.commit([poly])
        values, proof, _ = pcs.open(pdata, [pt], cheap_transcript(KB))
        self.assertTrue(bool(values[0] == eval_mle(poly, pt)))
        ok, _ = pcs.verify(comm, [pt], values, proof, cheap_transcript(KB))
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        poly = rand_field(3, (8,), KB)
        pt = rand_field(4, (3,), KB)
        pcs = DensePcs()
        comm, _ = pcs.commit([poly])
        bad = jnp.stack([eval_mle(poly, pt) + jnp.ones((), KB)])
        ok, _ = pcs.verify(comm, [pt], bad, None, cheap_transcript(KB))
        self.assertFalse(bool(ok))


class WitnessOpenGlueTest(absltest.TestCase):
    def _carry_for(
        self, inst: R1CS, z: Array, r_x: Array, r: Array, r_y: Array
    ) -> SpartanCarry:
        # Build the carry the glue reads, with a consistent inner_final.
        eval_abc = inst.eval_combined_matrix(r_x, r_y, r)
        z_eval = eval_mle(z, r_y)
        return SpartanCarry(r_x=r_x, r_batch=r, r_y=r_y, inner_final=eval_abc * z_eval)

    def test_glue_roundtrip_accepts(self) -> None:
        inst, z, _, io = toy_r1cs(5, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        nvp = inst.num_vars_padded
        pcs = DensePcs()
        comm, pdata = pcs.commit([z[:nvp]])
        r_x = rand_field(10, (inst.s_x,), KB)
        r = jnp.asarray(rand_field(11, (1,), KB)[0])
        r_y = rand_field(12, (inst.s_y,), KB)
        carry = self._carry_for(inst, z, r_x, r, r_y)
        _, _, msg = WitnessOpenProver(pcs, pdata)(carry, cheap_transcript(KB))
        _, _, ok = WitnessOpenVerifier(pcs, comm, inst, io)(
            carry, msg, cheap_transcript(KB)
        )
        self.assertTrue(bool(ok))

    def test_glue_rejects_wrong_inner_final(self) -> None:
        inst, z, _, io = toy_r1cs(6, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        nvp = inst.num_vars_padded
        pcs = DensePcs()
        comm, pdata = pcs.commit([z[:nvp]])
        r_x = rand_field(20, (inst.s_x,), KB)
        r = jnp.asarray(rand_field(21, (1,), KB)[0])
        r_y = rand_field(22, (inst.s_y,), KB)
        carry = self._carry_for(inst, z, r_x, r, r_y)
        bad = SpartanCarry(
            r_x=r_x,
            r_batch=r,
            r_y=r_y,
            inner_final=cast(Array, carry.inner_final) + jnp.ones((), KB),
        )
        _, _, msg = WitnessOpenProver(pcs, pdata)(carry, cheap_transcript(KB))
        _, _, ok = WitnessOpenVerifier(pcs, comm, inst, io)(
            bad, msg, cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))

    def test_public_half_reconstruction(self) -> None:
        # eval_public_half + recombine reproduce z̃(r_y) for the high half.
        _, z, _, io = toy_r1cs(7, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        nvp = 4
        r_y = rand_field(30, (3,), KB)
        eval_w = eval_mle(z[:nvp], r_y[1:])
        eval_pub = eval_public_half(io, r_y[1:], nvp)
        self.assertTrue(
            bool(recombine_z_eval(eval_w, eval_pub, r_y[0]) == eval_mle(z, r_y))
        )


if __name__ == "__main__":
    absltest.main()
