# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.poly.multilinear import eval_mle
from zorch.spartan.r1cs import (
    assignment,
    eval_public_half,
    recombine_z_eval,
)
from zorch.spartan.testing.toy import toy_r1cs
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear_mont


class R1CSTest(absltest.TestCase):
    def test_toy_is_satisfied(self) -> None:
        inst, z, _, _ = toy_r1cs(1, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        self.assertTrue(bool(inst.is_satisfied(z)))

    def test_dimensions(self) -> None:
        inst, _, _, _ = toy_r1cs(2, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        self.assertEqual(inst.num_cons, 8)
        self.assertEqual(inst.num_cols, 8)
        self.assertEqual(inst.s_x, 3)
        self.assertEqual(inst.s_y, 3)
        self.assertEqual(inst.num_vars_padded, 4)

    def test_combined_row_mle_matches_matrix_eval(self) -> None:
        # eval_mle(combined_row_mle(r_x, r), r_y) == eval_combined_matrix(r_x,r_y,r):
        # binding the row vars then the column vars equals the 2-var MLE.
        inst, _, _, _ = toy_r1cs(3, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        r_x = rand_field(40, (inst.s_x,), KB)
        r_y = rand_field(41, (inst.s_y,), KB)
        r = jnp.asarray(rand_field(42, (1,), KB)[0])
        row_poly = inst.combined_row_mle(r_x, r)
        got = eval_mle(row_poly, r_y)
        want = inst.eval_combined_matrix(r_x, r_y, r)
        self.assertTrue(bool(got == want))

    def test_z_eval_reconstructs_from_halves(self) -> None:
        # z̃(r_y) == (1-r_y0)·W̃(r_y[1:]) + r_y0·pub̃(r_y[1:]).
        _, z, witness, io = toy_r1cs(4, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        nvp = 4
        r_y = rand_field(43, (3,), KB)
        z_eval = eval_mle(z, r_y)
        eval_w = eval_mle(z[:nvp], r_y[1:])
        eval_pub = eval_public_half(io, r_y[1:], nvp)
        got = recombine_z_eval(eval_w, eval_pub, r_y[0])
        self.assertTrue(bool(got == z_eval))

    def test_assignment_layout(self) -> None:
        w = rand_field(50, (3,), KB)
        io = rand_field(51, (2,), KB)
        z = assignment(w, io, num_vars_padded=4, num_io=2)
        self.assertEqual(z.shape, (8,))
        self.assertTrue(bool(jnp.all(z[:3] == w)))
        self.assertTrue(bool(z[3] == jnp.zeros((), KB)))  # low-half pad
        self.assertTrue(bool(z[4] == jnp.ones((), KB)))  # constant 1
        self.assertTrue(bool(jnp.all(z[5:7] == io)))


if __name__ == "__main__":
    absltest.main()
