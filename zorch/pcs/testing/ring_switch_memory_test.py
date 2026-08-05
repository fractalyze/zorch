# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-only regression coverage for ring-switch GPU working memory."""

from __future__ import annotations

import frx
import frx.numpy as fnp
import zk_dtypes  # noqa: F401  (registers binary_field_ghash)
from absl.testing import absltest

from zorch.pcs.ring_switch import bit_slice_evals, rs_eq_ind
from zorch.utils.binary_field import byte_select_xor_reduce


class RingSwitchMemoryTest(absltest.TestCase):
    def test_large_shapes_do_not_materialize_bit_selection(self) -> None:
        if frx.default_backend() != "gpu":
            self.skipTest("compiled GPU memory analysis requires a GPU backend")

        dtype = fnp.binary_field_ghash
        shape = frx.ShapeDtypeStruct
        n = 1 << 24
        compiled = (
            bit_slice_evals.lower(shape((n,), dtype), shape((n,), dtype)).compile(),
            # Batched (n, N) share-the-witness path must stay bounded too — it
            # accumulates into (N, W, L), never the (n, N, W, L) broadcast.
            bit_slice_evals.lower(shape((n,), dtype), shape((n, 2), dtype)).compile(),
            rs_eq_ind.lower(shape((n,), dtype), shape((128,), dtype)).compile(),
            byte_select_xor_reduce.lower(
                shape((1 << 25, 8), fnp.uint8), shape((64,), dtype)
            ).compile(),
        )
        for executable in compiled:
            memory = executable.memory_analysis()
            self.assertLess(
                memory.temp_size_in_bytes,
                1 << 30,
                f"bit-select lowering uses {memory.temp_size_in_bytes} temporary bytes",
            )


if __name__ == "__main__":
    absltest.main()
