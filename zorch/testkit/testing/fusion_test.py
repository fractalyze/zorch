# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.testkit.fusion import assert_fusion_ready

KB = zk_dtypes.koalabear_mont


class AssertFusionReadyTest(absltest.TestCase):
    def test_straight_line_with_one_reduce_passes(self) -> None:
        x = jnp.arange(8, dtype=KB)
        assert_fusion_ready(lambda a: jnp.sum(a * a + a), x, reduces=1)

    def test_element_wise_no_reduce_passes(self) -> None:
        x = jnp.arange(8, dtype=KB)
        assert_fusion_ready(lambda a: a * a + a, x, reduces=0)

    def test_extra_reduce_fails(self) -> None:
        x = jnp.arange(8, dtype=KB)
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda a: jnp.sum(a) + jnp.sum(a * a), x, reduces=1)

    def test_boundary_op_gather_fails(self) -> None:
        x = jnp.arange(8, dtype=KB)
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda a: a[jnp.array([0, 2, 1, 3])], x, reduces=0)


if __name__ == "__main__":
    absltest.main()
