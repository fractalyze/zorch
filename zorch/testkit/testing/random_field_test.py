# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest

from zorch.testkit.random_field import rand_ext_field, rand_field

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


class RandFieldTest(absltest.TestCase):
    def test_shape_dtype_deterministic(self) -> None:
        a = rand_field(0, (8,), KB)
        b = rand_field(0, (8,), KB)
        self.assertEqual(a.shape, (8,))
        self.assertEqual(a.dtype, KB)
        self.assertTrue(bool(fnp.all(a == b)))  # same seed -> same values

    def test_different_seed_differs(self) -> None:
        a = rand_field(0, (16,), KB)
        b = rand_field(1, (16,), KB)
        self.assertFalse(bool(fnp.all(a == b)))

    def test_rejects_extension_dtype(self) -> None:
        # Casting an integer to an extension field embeds it into the base
        # subfield (higher coordinates zero) -- a degenerate draw. The guard
        # forces rand_ext_field, which draws a generic element.
        with self.assertRaisesRegex(ValueError, "extension field"):
            rand_field(0, (4,), EF)


class RandExtFieldTest(absltest.TestCase):
    def test_draws_generic_extension_elements(self) -> None:
        a = rand_ext_field(0, (4,), KB, EF)
        self.assertEqual(a.dtype, EF)
        # Generic: the higher extension coordinates are not all zero (which is
        # exactly what an integer-cast `rand_field(..., EF)` would produce).
        higher = np.asarray(a.view(KB)).reshape(-1, 4)[:, 1:]
        self.assertFalse(bool(np.all(higher == 0)))


if __name__ == "__main__":
    absltest.main()
