# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.poly.eq import (
    contract_hypercube_step,
    eq_factor,
    eq_root,
    eval_eq,
    expand_eq_to_hypercube,
    expand_hypercube_step,
)

KB = zk_dtypes.koalabear_mont


class ExpandEqTest(absltest.TestCase):
    def test_partition_of_unity(self) -> None:
        # Σ_w eq(w, x) == 1 for any x.
        x = jnp.array([2, 5, 7], dtype=KB)
        out = expand_eq_to_hypercube(x, jnp.ones([], dtype=KB))
        self.assertEqual(out.shape, (8,))
        self.assertEqual(int(out.sum()), 1)

    def test_msb_first_indexing(self) -> None:
        # result[nat(w)] with w[0] as MSB: x=[2,3] -> result[2]=eq([1,0],x)=2*(1-3).
        x = jnp.array([2, 3], dtype=KB)
        out = expand_eq_to_hypercube(x, jnp.ones([], dtype=KB))
        # 2*(1-3) = 2*(-2) = -4 ≡ p-4 in KB; compare via field arithmetic.
        expected = jnp.array(2, dtype=KB) * (
            jnp.array(1, dtype=KB) - jnp.array(3, dtype=KB)
        )
        self.assertEqual(int(out[2]), int(expected))


class EqFactorTest(absltest.TestCase):
    def test_matches_eval_eq_on_length_one_points(self) -> None:
        t = jnp.array(5, dtype=KB)
        z = jnp.array(9, dtype=KB)
        self.assertEqual(int(eq_factor(t, z)), int(eval_eq(t[None], z[None])))

    def test_selects_coordinate_at_boolean_t(self) -> None:
        # eq(0, z) = 1 - z and eq(1, z) = z: the factor a bound bit selects.
        z = jnp.array(7, dtype=KB)
        one = jnp.ones((), dtype=KB)
        self.assertEqual(int(eq_factor(jnp.zeros((), KB), z)), int(one - z))
        self.assertEqual(int(eq_factor(one, z)), int(z))

    def test_broadcasts_elementwise(self) -> None:
        t = jnp.array([0, 1, 3], dtype=KB)
        z = jnp.array([5, 5, 5], dtype=KB)
        out = eq_factor(t, z)
        self.assertEqual(out.shape, (3,))
        for i in range(3):
            self.assertEqual(int(out[i]), int(eq_factor(t[i], z[i])))


class EqRootTest(absltest.TestCase):
    def test_factor_vanishes_at_root(self) -> None:
        z = jnp.array(11, dtype=KB)
        self.assertEqual(int(eq_factor(eq_root(z), z)), 0)

    def test_broadcasts_over_points(self) -> None:
        z = jnp.array([3, 5, 7], dtype=KB)
        roots = eq_root(z)
        self.assertEqual(roots.shape, (3,))
        self.assertTrue(bool(jnp.all(eq_factor(roots, z) == jnp.zeros((3,), KB))))


class ContractHypercubeStepTest(absltest.TestCase):
    def test_sums_adjacent_pairs(self) -> None:
        state = jnp.array([1, 2, 3, 4, 5, 6], dtype=KB)
        out = contract_hypercube_step(state)
        self.assertTrue(bool(jnp.all(out == jnp.array([3, 7, 11], dtype=KB))))

    def test_marginalizes_the_lsb_variable(self) -> None:
        # Σ_b eq((w, b), x) = eq(w, x[:-1]): contracting the expanded table for
        # x recovers the table for x without its LSB coordinate, exactly.
        x = jnp.array([2, 5, 7], dtype=KB)
        one = jnp.ones((), dtype=KB)
        full = expand_eq_to_hypercube(x, one)
        want = expand_eq_to_hypercube(x[:-1], one)
        self.assertTrue(bool(jnp.all(contract_hypercube_step(full) == want)))

    def test_inverts_expand_step(self) -> None:
        # expand splits each entry into (1-coord)/coord shares; contracting
        # sums them back to the entry exactly.
        state = jnp.array([3, 9, 4, 6], dtype=KB)
        expanded = expand_hypercube_step(state, jnp.array(5, KB))
        self.assertTrue(bool(jnp.all(contract_hypercube_step(expanded) == state)))


class EvalEqTest(absltest.TestCase):
    def test_matches_hypercube_inner_product(self) -> None:
        # eq is the reproducing kernel: eq(w,x) == Σ_v eq(v,w)·eq(v,x).
        w = jnp.array([3, 5, 7], dtype=KB)
        x = jnp.array([2, 9, 4], dtype=KB)
        ew = expand_eq_to_hypercube(w, jnp.ones([], dtype=KB))
        ex = expand_eq_to_hypercube(x, jnp.ones([], dtype=KB))
        self.assertEqual(int(eval_eq(w, x)), int((ew * ex).sum()))

    def test_one_on_equal_boolean_points(self) -> None:
        a = jnp.array([1, 0, 1], dtype=KB)
        self.assertEqual(int(eval_eq(a, a)), 1)
        b = jnp.array([1, 1, 1], dtype=KB)
        self.assertEqual(int(eval_eq(a, b)), 0)


if __name__ == "__main__":
    absltest.main()
