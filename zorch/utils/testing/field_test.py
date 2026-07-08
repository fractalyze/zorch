# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.utils.field import base_field, naturals

KB = zk_dtypes.koalabear_mont
KX = zk_dtypes.koalabearx4_mont


class BaseFieldTest(absltest.TestCase):
    def test_prime_field_is_itself(self) -> None:
        self.assertEqual(base_field(KB), KB)

    def test_extension_gives_its_prime_subfield(self) -> None:
        self.assertEqual(base_field(KX), KB)


class NaturalsTest(absltest.TestCase):
    def test_values_and_base_dtype(self) -> None:
        # [0..n−1] in the base field, whether asked for the base or an extension.
        for dt in (KB, KX):
            got = naturals(5, dt)
            self.assertEqual(got.dtype, jnp.dtype(KB))
            self.assertEqual([int(v) for v in got], [0, 1, 2, 3, 4])

    def test_matches_per_element_embedding(self) -> None:
        # The base nodes an extension caller promotes must equal embedding each
        # integer into the extension directly (the pattern this replaces).
        got = naturals(6, KX).astype(KX)
        want = jnp.stack([jnp.array(i, KX) for i in range(6)])
        self.assertTrue(bool(jnp.all(got == want)))


if __name__ == "__main__":
    absltest.main()
