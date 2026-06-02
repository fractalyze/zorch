# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from absl.testing import absltest

from zorch.utils.bits import is_power_of_two, log2_strict_usize


class BitsTest(absltest.TestCase):
    def test_is_power_of_two(self):
        self.assertTrue(is_power_of_two(1))
        self.assertTrue(is_power_of_two(8))
        self.assertFalse(is_power_of_two(0))
        self.assertFalse(is_power_of_two(6))

    def test_log2_strict_usize(self):
        self.assertEqual(log2_strict_usize(1), 0)
        self.assertEqual(log2_strict_usize(8), 3)

    def test_log2_strict_rejects_non_pow2(self):
        with self.assertRaises(ValueError):
            log2_strict_usize(6)


if __name__ == "__main__":
    absltest.main()
