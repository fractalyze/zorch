# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import frx.numpy as jnp
import numpy as np
import zk_dtypes  # noqa: F401  (registers the binary_field_* dtypes)
from absl.testing import absltest, parameterized
from frx import Array, lax

from zorch.utils.binary_field import _to_limbs, field_bit_width, pack, unpack

_DTYPES = (jnp.binary_field_ghash, jnp.binary_field_t7)


def _rand_field(dtype: Any, n: int, seed: int) -> Array:
    """Random field elements from random uint32 storage limbs (no field RNG)."""
    n_lanes = jnp.dtype(dtype).itemsize // 4
    raw = np.random.default_rng(seed).integers(
        0, 1 << 32, size=(n, n_lanes), dtype=np.uint32
    )
    return lax.bitcast_convert_type(jnp.asarray(raw), dtype)


def _coeff_u8(coeffs: Array) -> np.ndarray:
    """F_2 (t0) coefficients as a {0,1} uint8 numpy array (host byte view; t0 is
    stored one byte per element as 0x00 / 0x01)."""
    return np.asarray(coeffs).view(np.uint8)


class BinaryFieldReprTest(parameterized.TestCase):
    @parameterized.parameters(*_DTYPES)
    def test_bit_width(self, dtype: Any) -> None:
        self.assertEqual(field_bit_width(dtype), jnp.dtype(dtype).itemsize * 8)

    @parameterized.parameters(*_DTYPES)
    def test_unpack_is_f2_vector(self, dtype: Any) -> None:
        x = _rand_field(dtype, 17, 0)
        coeffs = unpack(x)
        self.assertEqual(jnp.dtype(coeffs.dtype), jnp.dtype(jnp.binary_field_t0))
        self.assertEqual(coeffs.shape, (17, field_bit_width(dtype)))
        self.assertTrue(bool(np.all(np.isin(_coeff_u8(coeffs), (0, 1)))))

    @parameterized.parameters(*_DTYPES)
    def test_pack_unpack_roundtrip(self, dtype: Any) -> None:
        x = _rand_field(dtype, 17, 1)
        back = pack(unpack(x), dtype)
        np.testing.assert_array_equal(
            np.asarray(_to_limbs(back)), np.asarray(_to_limbs(x))
        )

    def test_bit_width_rejects_sub_lane_tower(self) -> None:
        narrow = [
            d
            for k in range(8)
            if (d := getattr(jnp, f"binary_field_t{k}", None)) is not None
            and jnp.dtype(d).itemsize * 8 < 32
        ]
        if not narrow:
            self.skipTest("no sub-32-bit binary tower dtype available")
        with self.assertRaises(ValueError):
            field_bit_width(narrow[0])


if __name__ == "__main__":
    absltest.main()
