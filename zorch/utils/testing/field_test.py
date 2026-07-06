# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for zorch.utils.field: is_binary_field, field_sum.

The binary cases run on GPU only: `binary_field_ghash` arithmetic is unlowered on
the CPU PJRT path in the ZKX env, and the reduce-add that `field_sum` routes
around SIGSEGVs specifically on the CUDA backend (zorch#400) — so the CUDA path is
both the thing under test and the only place the ghash draws are non-vacuous.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # binary_field_ghash is uint64-backed

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import zk_dtypes  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from jax import Array, lax  # noqa: E402

from zorch.utils.field import field_sum, is_binary_field  # noqa: E402

KB = zk_dtypes.koalabear_mont  # prime field
GH = zk_dtypes.binary_field_ghash  # binary field GF(2^128)

_on_gpu = absltest.skipUnless(
    jax.default_backend() == "gpu",
    "binary_field_ghash reduce-add is a CUDA-only lowering gap (zorch#400)",
)


def _gh(vals: list[int]) -> Array:
    return jnp.array(vals, dtype=GH)


def _u64(a: Array) -> np.ndarray:  # packed 128-bit rep as (..., 2) uint64 lanes
    return np.asarray(lax.bitcast_convert_type(a, jnp.uint64))


class IsBinaryFieldTest(absltest.TestCase):
    def test_binary_is_true(self) -> None:
        self.assertTrue(is_binary_field(GH))

    def test_prime_is_false(self) -> None:
        self.assertFalse(is_binary_field(KB))


class FieldSumPrimeTest(parameterized.TestCase):
    """Prime dtypes take the native `jnp.sum`; pin the passthrough contract."""

    @parameterized.named_parameters(
        dict(testcase_name="1d_full", shape=(8,), axis=None),
        dict(testcase_name="1d_axis0", shape=(7,), axis=0),
        dict(testcase_name="2d_axis0", shape=(5, 4), axis=0),
        dict(testcase_name="2d_axis1", shape=(3, 4), axis=1),
    )
    def test_matches_jnp_sum(self, shape: tuple[int, ...], axis: int | None) -> None:
        ints = np.random.default_rng(0).integers(0, 1 << 30, size=shape, dtype=np.int64)
        x = jnp.asarray(ints, dtype=KB)
        got = field_sum(x, axis=axis)
        self.assertTrue(bool(jnp.all(got == jnp.sum(x, axis=axis))))


class FieldSumBinaryTest(parameterized.TestCase):
    """GF(2^128) addition is a bitwise XOR of the 128-bit representation, so a
    host XOR-reduce over the packed uint64 lanes is an exact, reduce-free oracle
    (independent of the field encoding and of the folding order under test)."""

    @_on_gpu
    @parameterized.named_parameters(
        dict(testcase_name="1d_even_full", shape=(8,), axis=None),
        dict(testcase_name="1d_odd_full", shape=(5,), axis=None),
        dict(testcase_name="1d_odd_axis0", shape=(7,), axis=0),
        dict(testcase_name="2d_axis0_oddlead", shape=(5, 4), axis=0),
        dict(testcase_name="2d_axis1", shape=(3, 4), axis=1),
        dict(testcase_name="2d_axis0_single", shape=(1, 4), axis=0),
    )
    def test_matches_xor_reduce(self, shape: tuple[int, ...], axis: int | None) -> None:
        n = int(np.prod(shape))
        x = _gh(list(range(1, n + 1))).reshape(shape)  # distinct nonzero draws
        got = _u64(field_sum(x, axis=axis))
        red = tuple(range(len(shape))) if axis is None else axis
        want = np.bitwise_xor.reduce(_u64(x), axis=red)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(bool(np.array_equal(got, want)))


if __name__ == "__main__":
    absltest.main()
