# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.testkit.fusion import assert_fusion_ready, assert_marker_recognized
from zorch.testkit.koalabear16 import koalabear16_perm

KB = zk_dtypes.koalabear_mont


class AssertFusionReadyTest(absltest.TestCase):
    def test_straight_line_with_one_reduce_passes(self) -> None:
        x = fnp.arange(8, dtype=KB)
        assert_fusion_ready(lambda a: fnp.sum(a * a + a), x, reduces=1)

    def test_element_wise_no_reduce_passes(self) -> None:
        x = fnp.arange(8, dtype=KB)
        assert_fusion_ready(lambda a: a * a + a, x, reduces=0)

    def test_extra_reduce_fails(self) -> None:
        x = fnp.arange(8, dtype=KB)
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda a: fnp.sum(a) + fnp.sum(a * a), x, reduces=1)

    def test_boundary_op_gather_fails(self) -> None:
        x = fnp.arange(8, dtype=KB)
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda a: a[fnp.array([0, 2, 1, 3])], x, reduces=0)


class AssertMarkerRecognizedTest(absltest.TestCase):
    def test_routed_marker_passes(self) -> None:
        perm = koalabear16_perm()
        assert_marker_recognized("poseidon2", perm.permute, fnp.arange(16, dtype=KB))

    def test_wrong_routing_key_fails(self) -> None:
        # `poseidon` is a prefix of `poseidon2`: a substring match would let the
        # Poseidon2 emitter satisfy the classic Poseidon assertion.
        perm = koalabear16_perm()
        with self.assertRaises(AssertionError):
            assert_marker_recognized("poseidon", perm.permute, fnp.arange(16, dtype=KB))

    def test_unmarked_body_fails(self) -> None:
        x = fnp.arange(8, dtype=KB)
        with self.assertRaises(AssertionError):
            assert_marker_recognized("poseidon2", lambda a: a * a + a, x)


if __name__ == "__main__":
    absltest.main()
