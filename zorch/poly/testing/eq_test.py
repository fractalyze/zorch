# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube

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
