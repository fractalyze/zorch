# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes  # noqa: F401  (registers the binary_field_* dtypes)
from absl.testing import absltest, parameterized
from frx import Array, lax

from zorch.utils.binary_field import (
    _to_limbs,
    bit_select_xor_reduce,
    byte_select_xor_reduce,
    byte_slice_xor_reduce,
    field_bit_width,
    pack,
    unpack,
)

_DTYPES = (fnp.binary_field_ghash, fnp.binary_field_t7)


def _rand_field(dtype: Any, n: int, seed: int) -> Array:
    """Random field elements from random uint32 storage limbs (no field RNG)."""
    n_lanes = fnp.dtype(dtype).itemsize // 4
    raw = np.random.default_rng(seed).integers(
        0, 1 << 32, size=(n, n_lanes), dtype=np.uint32
    )
    return lax.bitcast_convert_type(fnp.asarray(raw), dtype)


def _coeff_u8(coeffs: Array) -> np.ndarray:
    """F_2 (t0) coefficients as a {0,1} uint8 numpy array (host byte view; t0 is
    stored one byte per element as 0x00 / 0x01)."""
    return np.asarray(coeffs).view(np.uint8)


class BinaryFieldReprTest(parameterized.TestCase):
    @parameterized.parameters(*_DTYPES)
    def test_bit_width(self, dtype: Any) -> None:
        self.assertEqual(field_bit_width(dtype), fnp.dtype(dtype).itemsize * 8)

    @parameterized.parameters(*_DTYPES)
    def test_unpack_is_f2_vector(self, dtype: Any) -> None:
        x = _rand_field(dtype, 17, 0)
        coeffs = unpack(x)
        self.assertEqual(fnp.dtype(coeffs.dtype), fnp.dtype(fnp.binary_field_t0))
        self.assertEqual(coeffs.shape, (17, field_bit_width(dtype)))
        self.assertTrue(bool(np.all(np.isin(_coeff_u8(coeffs), (0, 1)))))

    @parameterized.parameters(*_DTYPES)
    def test_pack_unpack_roundtrip(self, dtype: Any) -> None:
        x = _rand_field(dtype, 17, 1)
        back = pack(unpack(x), dtype)
        np.testing.assert_array_equal(
            np.asarray(_to_limbs(back)), np.asarray(_to_limbs(x))
        )

    @parameterized.parameters(*_DTYPES)
    def test_bit_select_xor_reduce_axes_are_dual(self, dtype: Any) -> None:
        width = field_bit_width(dtype)
        selectors = _rand_field(dtype, 11, 2)
        element_values = _rand_field(dtype, 11, 3)
        bit_values = _rand_field(dtype, width, 4)

        by_element = bit_select_xor_reduce(selectors, element_values, reduce="elements")
        by_bit = bit_select_xor_reduce(selectors, bit_values, reduce="bits")
        bits = np.unpackbits(
            np.asarray(selectors).view(np.uint8).reshape(11, -1),
            axis=1,
            bitorder="little",
        )
        element_limbs = np.asarray(_to_limbs(element_values))
        bit_limbs = np.asarray(_to_limbs(bit_values))
        want_by_element = np.zeros((width, element_limbs.shape[1]), np.uint32)
        want_by_bit = np.zeros((11, bit_limbs.shape[1]), np.uint32)
        for i in range(11):
            for bit in range(width):
                if bits[i, bit]:
                    want_by_element[bit] ^= element_limbs[i]
                    want_by_bit[i] ^= bit_limbs[bit]
        np.testing.assert_array_equal(
            np.asarray(_to_limbs(by_element)), want_by_element
        )
        np.testing.assert_array_equal(np.asarray(_to_limbs(by_bit)), want_by_bit)

    @parameterized.parameters(*_DTYPES)
    def test_bit_select_xor_reduce_accepts_unpacked_rows(self, dtype: Any) -> None:
        rows = fnp.asarray(
            np.random.default_rng(5).integers(0, 2, size=(13, 64), dtype=np.uint8)
        )
        values = _rand_field(dtype, 64, 6)
        got = bit_select_xor_reduce(rows, values, reduce="bits")

        value_limbs = np.asarray(_to_limbs(values))
        want = np.zeros((13, value_limbs.shape[1]), np.uint32)
        for i in range(13):
            for bit in range(64):
                if rows[i, bit]:
                    want[i] ^= value_limbs[bit]
        np.testing.assert_array_equal(np.asarray(_to_limbs(got)), want)

    @parameterized.parameters(*_DTYPES)
    def test_byte_select_xor_reduce_accepts_packed_rows(self, dtype: Any) -> None:
        rng = np.random.default_rng(7)
        rows = rng.integers(0, 256, size=(13, 8), dtype=np.uint8)
        values = _rand_field(dtype, 64, 8)
        got = byte_select_xor_reduce(fnp.asarray(rows), values)

        unpacked = np.unpackbits(rows, axis=1, bitorder="little")
        value_limbs = np.asarray(_to_limbs(values))
        want = np.zeros((13, value_limbs.shape[1]), np.uint32)
        for i in range(13):
            for bit in range(64):
                if unpacked[i, bit]:
                    want[i] ^= value_limbs[bit]
        np.testing.assert_array_equal(np.asarray(_to_limbs(got)), want)

    @parameterized.parameters(
        (dtype, n, cols) for dtype in _DTYPES for n, cols in ((1, 1), (37, 5), (289, 3))
    )
    def test_byte_slice_xor_reduce_matches_reference(
        self, dtype: Any, n: int, cols: int
    ) -> None:
        rng = np.random.default_rng(11)
        selectors = rng.integers(0, 256, size=(n, cols), dtype=np.uint8)
        values = _rand_field(dtype, n, 12)
        got = byte_slice_xor_reduce(fnp.asarray(selectors), values)

        value_limbs = np.asarray(_to_limbs(values))
        want = np.zeros((cols, 8, value_limbs.shape[1]), np.uint32)
        for i in range(n):
            for c in range(cols):
                for bit in range(8):
                    if (selectors[i, c] >> bit) & 1:
                        want[c, bit] ^= value_limbs[i]
        self.assertEqual(got.shape, (cols, 8))
        np.testing.assert_array_equal(
            np.asarray(_to_limbs(got)).reshape(cols, 8, -1), want
        )

    def test_byte_slice_xor_reduce_validates(self) -> None:
        sel = fnp.asarray(np.zeros((4, 2), np.uint8))
        vals = _rand_field(fnp.binary_field_ghash, 4, 13)
        with self.assertRaises(ValueError):
            byte_slice_xor_reduce(sel[:, 0], vals)  # 1D selectors
        with self.assertRaises(ValueError):
            byte_slice_xor_reduce(sel.astype(fnp.int32), vals)  # non-u8
        with self.assertRaises(ValueError):
            byte_slice_xor_reduce(sel, vals[:3])  # length mismatch
        with self.assertRaises(ValueError):
            byte_slice_xor_reduce(sel[:0], vals[:0])  # empty
        with self.assertRaises(ValueError):
            byte_slice_xor_reduce(sel, fnp.asarray(np.zeros(4, np.uint32)))  # non-field

    def test_bit_width_rejects_sub_lane_tower(self) -> None:
        narrow = [
            d
            for k in range(8)
            if (d := getattr(fnp, f"binary_field_t{k}", None)) is not None
            and fnp.dtype(d).itemsize * 8 < 32
        ]
        if not narrow:
            self.skipTest("no sub-32-bit binary tower dtype available")
        with self.assertRaises(ValueError):
            field_bit_width(narrow[0])


if __name__ == "__main__":
    absltest.main()
