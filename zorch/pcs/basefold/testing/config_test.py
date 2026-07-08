# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BasefoldConfig fold-schedule knobs: the zorch-native default and a
non-default (row-batch-prefix + multi-arity) schedule."""
from __future__ import annotations

from absl.testing import absltest

from zorch.pcs.basefold.config import BasefoldConfig


class ConfigTest(absltest.TestCase):
    def test_zorch_native_default_commits_per_round(self) -> None:
        c = BasefoldConfig(num_vars=4, num_queries=4)
        self.assertTrue(c.commits_per_round)
        self.assertEqual(c.row_batch_prefix, 0)
        self.assertEqual(c.fold_arities, ())

    def test_flock_shaped_schedule(self) -> None:
        c = BasefoldConfig(
            num_vars=9, num_queries=8, row_batch_prefix=5, fold_arities=(2, 2)
        )
        self.assertFalse(c.commits_per_round)

    def test_cadence_schedule_must_cover_num_vars(self) -> None:
        # prefix 5 + sum(2, 2) = 9 != num_vars 8 -> reject at construction.
        with self.assertRaisesRegex(ValueError, "cadence fold schedule"):
            BasefoldConfig(
                num_vars=8, num_queries=8, row_batch_prefix=5, fold_arities=(2, 2)
            )


if __name__ == "__main__":
    absltest.main()
