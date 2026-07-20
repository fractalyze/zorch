# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx

frx.config.update("jax_enable_x64", True)  # binary_field_ghash is uint64-backed

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
import zk_dtypes  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from frx import Array, lax  # noqa: E402

from zorch.utils.field import (
    base_field,
    is_binary_field,
    join_limbs,
    naturals,
    split_limbs,
)

KB = zk_dtypes.koalabear_mont
KX = zk_dtypes.koalabearx4_mont
GH = zk_dtypes.binary_field_ghash


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
            self.assertEqual(got.dtype, fnp.dtype(KB))
            self.assertEqual([int(v) for v in got], [0, 1, 2, 3, 4])

    def test_matches_per_element_embedding(self) -> None:
        # The base nodes an extension caller promotes must equal embedding each
        # integer into the extension directly (the pattern this replaces).
        got = naturals(6, KX).astype(KX)
        want = fnp.stack([fnp.array(i, KX) for i in range(6)])
        self.assertTrue(bool(fnp.all(got == want)))


class IsBinaryFieldTest(absltest.TestCase):
    def test_binary_is_true(self) -> None:
        self.assertTrue(is_binary_field(GH))

    def test_prime_and_extension_are_false(self) -> None:
        self.assertFalse(is_binary_field(KB))
        self.assertFalse(is_binary_field(KX))


def _gh(n: int, seed: int = 0) -> Array:
    """`n` GF(2^128) elements from random uint32 limbs (the ghash list constructor
    is vacuous on the CPU PJRT path; a limb bitcast is not)."""
    raw = np.random.default_rng(seed).integers(0, 1 << 32, size=(n, 4), dtype=np.uint32)
    return lax.bitcast_convert_type(fnp.asarray(raw), GH)


def _u64(a: Array) -> np.ndarray:
    """The packed 128-bit representation as `(..., 2)` uint64 lanes."""
    return np.asarray(lax.bitcast_convert_type(a, fnp.uint64))


class LimbViewTest(absltest.TestCase):
    """`split_limbs` / `join_limbs`: the view between an extension array and
    its base limbs."""

    def _ext(self, shape: tuple[int, ...], start: int = 1) -> Array:
        n = int(np.prod(shape)) * 4  # KX is degree 4
        limbs = fnp.array(np.arange(start, start + n, dtype=np.uint64), dtype=KB)
        return join_limbs(limbs.reshape(*shape, 4), KX)

    def test_round_trip_recovers_the_elements(self) -> None:
        values = self._ext((7,))
        rows = split_limbs(values)
        self.assertEqual(rows.shape, (7, 4))
        self.assertEqual(rows.dtype, fnp.dtype(KB))
        self.assertTrue(fnp.array_equal(join_limbs(rows, KX), values))

    def test_each_row_is_one_element(self) -> None:
        # The layout every caller depends on: row i holds element i's
        # coefficients, not a stride across the array.
        values = self._ext((4,))
        rows = split_limbs(values)
        for i in range(4):
            one = join_limbs(rows[i : i + 1], KX)
            self.assertTrue(fnp.array_equal(one.reshape(()), values[i]))

    def test_degree_comes_from_the_dtype(self) -> None:
        # The point of the helper: a caller writing `.reshape(n, 4)` by hand is
        # pinned to a degree-4 field, this is not.
        self.assertEqual(split_limbs(self._ext((3,))).shape[-1], 4)

    def test_leading_axes_are_preserved(self) -> None:
        self.assertEqual(split_limbs(self._ext((2, 5))).shape, (2, 5, 4))

    def test_a_base_array_passes_through(self) -> None:
        # Not (n, 1): a base element is one coefficient, and a length-1 axis
        # would imply an extension that is not there.
        values = naturals(6, KB)
        self.assertTrue(fnp.array_equal(split_limbs(values), values))
        self.assertTrue(fnp.array_equal(join_limbs(values, KB), values))

    def test_rejects_a_trailing_axis_that_is_not_the_degree(self) -> None:
        for bad in (3, 5, 8):
            with self.subTest(trailing=bad):
                limbs = naturals(bad, KB).reshape(1, bad)
                with self.assertRaisesRegex(ValueError, "must be the degree"):
                    join_limbs(limbs, KX)


class BinaryFieldReduceAddTest(parameterized.TestCase):
    """GF(2^128) addition is a bitwise XOR of the 128-bit representation, so a host
    XOR-reduce over the packed uint64 lanes is an exact, reduce-free oracle for the
    native `fnp.sum`. This pins the binary-field reduce-add lowering that
    `ring_switch.inner_product` — and any field reduction — relies on: a wheel that
    regressed it (cf. the historical CUDA reduce-add SIGSEGV, zorch#400) would fail
    here rather than deep in a prover. Run under `jit`, where the full-reduction
    (0-d) case lowers cleanly."""

    @parameterized.named_parameters(
        dict(testcase_name="1d_even_full", shape=(8,), axis=None),
        dict(testcase_name="1d_odd_full", shape=(5,), axis=None),
        dict(testcase_name="1d_odd_axis0", shape=(7,), axis=0),
        dict(testcase_name="2d_axis0_oddlead", shape=(5, 4), axis=0),
        dict(testcase_name="2d_axis1", shape=(3, 4), axis=1),
        dict(testcase_name="2d_axis0_single", shape=(1, 4), axis=0),
    )
    def test_jnp_sum_matches_xor_reduce(
        self, shape: tuple[int, ...], axis: int | None
    ) -> None:
        n = int(np.prod(shape))
        x = _gh(n).reshape(shape)
        got = _u64(frx.jit(lambda a: fnp.sum(a, axis=axis))(x))
        red = tuple(range(len(shape))) if axis is None else axis
        want = np.bitwise_xor.reduce(_u64(x), axis=red)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(bool(np.array_equal(got, want)))


if __name__ == "__main__":
    absltest.main()
