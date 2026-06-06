# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.testkit.random_field import rand_field


class RandFieldTest(absltest.TestCase):
    def test_shape_dtype_deterministic(self) -> None:
        a = rand_field(0, (8,), zk_dtypes.koalabear_mont)
        b = rand_field(0, (8,), zk_dtypes.koalabear_mont)
        self.assertEqual(a.shape, (8,))
        self.assertEqual(a.dtype, zk_dtypes.koalabear_mont)
        self.assertTrue(bool(jnp.all(a == b)))  # same seed -> same values

    def test_different_seed_differs(self) -> None:
        a = rand_field(0, (16,), zk_dtypes.koalabear_mont)
        b = rand_field(1, (16,), zk_dtypes.koalabear_mont)
        self.assertFalse(bool(jnp.all(a == b)))


if __name__ == "__main__":
    absltest.main()
