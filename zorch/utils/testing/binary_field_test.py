# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes  # noqa: F401  (registers the binary_field_* dtypes)
from absl.testing import absltest, parameterized
from frx import Array, lax

from zorch.utils.binary_field import (
    _ELEMENTS_TARGET_PROGRAMS,
    _to_limbs,
    bit_select_xor_reduce,
    byte_select_xor_reduce,
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


def _selector_bits(selectors: Array, n: int) -> np.ndarray:
    """`selectors`' bits as a {0,1} `(n, W)` array, LSB-first — the host-side view
    of what `bit_select_xor_reduce` selects on."""
    return np.unpackbits(
        np.asarray(selectors).view(np.uint8).reshape(n, -1), axis=1, bitorder="little"
    )


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
        bits = _selector_bits(selectors, 11)
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
    def test_bit_select_xor_reduce_elements_batches_over_shared_selectors(
        self, dtype: Any
    ) -> None:
        """`reduce="elements"` accepts a selectors-major `(n, N)` values stack
        against `(n,)` selectors and returns `(N, W)`: row `k` equals the
        single-claim reduction against column `values[:, k]`. The ring-switch
        batched-open path reads the shared selectors (the packed witness) once
        for all `N` claims."""
        width = field_bit_width(dtype)
        n, batch = 11, 3
        selectors = _rand_field(dtype, n, 20)
        values = _rand_field(dtype, n * batch, 21).reshape(n, batch)

        batched = bit_select_xor_reduce(selectors, values, reduce="elements")
        self.assertEqual(batched.shape, (batch, width))

        for k in range(batch):
            one = bit_select_xor_reduce(selectors, values[:, k], reduce="elements")
            np.testing.assert_array_equal(
                np.asarray(_to_limbs(batched[k])), np.asarray(_to_limbs(one))
            )

    @parameterized.parameters(*_DTYPES)
    def test_bit_select_xor_reduce_elements_spans_multiple_tiles(
        self, dtype: Any
    ) -> None:
        """The `elements` GPU kernel walks `n` in a serial loop and zero-pads the
        tail, neither of which the 11-row shapes above reach — they fit in one
        iteration with nothing to pad.

        Size off `_ELEMENTS_TARGET_PROGRAMS` rather than a literal so the case
        survives a grid retune. The 4x covers a row tile up to 4 (the kernels
        pick their own `block`, so it is not importable); the odd remainder is
        what forces the tail padding."""
        n = 4 * _ELEMENTS_TARGET_PROGRAMS + 17

        selectors = _rand_field(dtype, n, 8)
        values = _rand_field(dtype, n, 9)

        got = bit_select_xor_reduce(selectors, values, reduce="elements")

        bits = _selector_bits(selectors, n)
        limbs = np.asarray(_to_limbs(values))
        want = np.bitwise_xor.reduce(
            np.where(bits[..., None], limbs[:, None, :], np.uint32(0)), axis=0
        )
        np.testing.assert_array_equal(np.asarray(_to_limbs(got)), want)

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
