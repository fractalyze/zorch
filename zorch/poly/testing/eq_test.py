# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import jax.numpy as jnp
import zk_dtypes

from zorch.poly.eq import expand_eq_to_hypercube

KB = zk_dtypes.koalabear


class ExpandEqTest(unittest.TestCase):
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
