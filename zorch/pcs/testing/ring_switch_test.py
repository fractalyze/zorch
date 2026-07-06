# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ring-switch reduction: bit kernels against pure-numpy bit-loop oracles, the
u/v duality identity that makes the reduction sound, and the succinct verifier
evaluation against the dense materialize-and-fold path.

The oracles never touch the kernels' lane algebra: they re-derive every bit
from the numpy byte buffer and accumulate with host XOR, so a lane-order or
reshape bug in the implementation cannot cancel itself out in the comparison.
The duality and dense-vs-succinct checks run on random *non-eq* tensors too —
the identities are bilinear, so random vectors are the stronger test.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.pcs.ring_switch import (
    RingSwitch,
    add,
    bit_slice_evals,
    eval_rs_eq,
    field_bit_width,
    inner_product,
    reduce_bit_claim,
    rs_eq_ind,
    tensor_algebra_transpose,
)

FIELDS = {
    "ghash": zk_dtypes.binary_field_ghash,
    "tower_t7": zk_dtypes.binary_field_t7,
    "tower_t6": zk_dtypes.binary_field_t6,  # W=64: keeps the kernels width-generic
}


def _rand(seed: int, shape: tuple[int, ...], dtype: Any) -> Array:
    """Uniform field elements drawn as raw storage bytes (every bit pattern is
    an element of a binary field)."""
    lanes = np.dtype(dtype).itemsize // 4
    raw = (
        np.random.default_rng(seed)
        .integers(0, 1 << 32, size=(*shape, lanes), dtype=np.uint64)
        .astype(np.uint32)
    )
    return jnp.asarray(raw.view(np.dtype(dtype)).reshape(shape))


def _np_bits(x: Array) -> np.ndarray:
    """(n,) field -> (n, W) 0/1 via the numpy byte buffer, little-endian."""
    b = np.asarray(x).view(np.uint8)
    n = x.shape[0] if x.shape else 1
    return np.unpackbits(b.reshape(n, -1), axis=1, bitorder="little")


def _np_lanes(x: Array) -> np.ndarray:
    flat = np.asarray(x).reshape(-1)  # 0-d (scalar claims) can't view in place
    return flat.view(np.uint32).reshape(*x.shape, -1)


def _np_field(lanes: np.ndarray, dtype: Any) -> np.ndarray:
    return lanes.astype(np.uint32).view(np.dtype(dtype)).reshape(lanes.shape[:-1])


class RingSwitchKernelsTest(parameterized.TestCase):
    """Each kernel against a from-scratch numpy bit loop."""

    @parameterized.named_parameters(FIELDS.items())
    def test_bit_slice_evals_matches_bit_loop(self, dtype: Any) -> None:
        w = field_bit_width(dtype)
        witness, tensor = _rand(1, (8,), dtype), _rand(2, (8,), dtype)
        bits, t_lanes = _np_bits(witness), _np_lanes(tensor)
        want = np.zeros((w, t_lanes.shape[-1]), dtype=np.uint32)
        for i in range(8):
            for r in range(w):
                if bits[i, r]:
                    want[r] ^= t_lanes[i]
        got = bit_slice_evals(witness, tensor)
        np.testing.assert_array_equal(_np_lanes(got), want)

    @parameterized.named_parameters(FIELDS.items())
    def test_rs_eq_ind_matches_bit_loop(self, dtype: Any) -> None:
        w = field_bit_width(dtype)
        tensor, eq_r = _rand(3, (8,), dtype), _rand(4, (w,), dtype)
        bits, eq_lanes = _np_bits(tensor), _np_lanes(eq_r)
        want = np.zeros((8, eq_lanes.shape[-1]), dtype=np.uint32)
        for i in range(8):
            for b in range(w):
                if bits[i, b]:
                    want[i] ^= eq_lanes[b]
        got = rs_eq_ind(tensor, eq_r)
        np.testing.assert_array_equal(_np_lanes(got), want)

    @parameterized.named_parameters(FIELDS.items())
    def test_transpose_is_the_bit_matrix_transpose(self, dtype: Any) -> None:
        w = field_bit_width(dtype)
        v = _rand(5, (w,), dtype)
        got = tensor_algebra_transpose(v)
        np.testing.assert_array_equal(_np_bits(got), _np_bits(v).T)
        back = tensor_algebra_transpose(got)
        np.testing.assert_array_equal(_np_lanes(back), _np_lanes(v))

    def test_add_is_xor(self) -> None:
        dtype = zk_dtypes.binary_field_ghash
        a, b = _rand(6, (4,), dtype), _rand(7, (4,), dtype)
        np.testing.assert_array_equal(_np_lanes(add(a, b)), _np_lanes(a) ^ _np_lanes(b))


class RingSwitchReductionTest(parameterized.TestCase):
    """The identity that makes the reduction work: both readings of
    `Σ_i tensor[i] ⊗ witness[i]` agree after the vertical fold, i.e.

        Σ_i witness[i] · rs_eq_ind[i]  ==  ⟨transpose(s_hat_v), eq(r'')⟩

    for *arbitrary* tensor and eq vectors (bilinearity — no eq structure
    assumed), in every supported representation.
    """

    @parameterized.named_parameters(FIELDS.items())
    def test_uv_duality(self, dtype: Any) -> None:
        w = field_bit_width(dtype)
        witness = _rand(11, (16,), dtype)
        tensor = _rand(12, (16,), dtype)
        eq_r = _rand(13, (w,), dtype)
        rs = reduce_bit_claim(witness, tensor, eq_r)
        self.assertIsInstance(rs, RingSwitch)
        lhs = inner_product(witness, rs.rs_eq_ind)
        np.testing.assert_array_equal(_np_lanes(lhs), _np_lanes(rs.claim))


class EvalRsEqTest(parameterized.TestCase):
    """Succinct verifier path vs dense: materialize `rs_eq_ind` from the eq
    tensor of `z` and fold it at `query`, then compare with `eval_rs_eq`."""

    @staticmethod
    def _eq_tensor(point: Array) -> Array:
        """eq(point, ·) over the hypercube, variable `i` at index bit `i`
        (LSB-first) — built by the textbook doubling, sharing no module code."""
        dtype = point.dtype
        lanes = np.dtype(dtype).itemsize // 4
        one_lanes = np.zeros((1, lanes), dtype=np.uint32)
        one_lanes[0, 0] = 1
        t = jnp.asarray(_np_field(one_lanes, dtype))
        one = t[0]
        for i in range(point.shape[0]):
            hi = t * point[i]
            lo = t * add(one, point[i])  # 1 - r = 1 + r in characteristic 2
            t = jnp.concatenate([lo, hi])
        return t

    @classmethod
    def _mle_eval(cls, evals: Array, point: Array) -> Array:
        """Fold variable 0 (LSB) first — the order `_eq_tensor` lays out."""
        v = evals
        one_like = cls._eq_tensor(point[:0])[0]
        for i in range(point.shape[0]):
            q = point[i]
            v = add(v[0::2] * add(one_like, q), v[1::2] * q)
        return v[0]

    @parameterized.named_parameters(FIELDS.items())
    def test_matches_dense_fold(self, dtype: Any) -> None:
        w = field_bit_width(dtype)
        ell = 3
        z_vals = _rand(21, (ell,), dtype)
        query = _rand(22, (ell,), dtype)
        eq_r = _rand(23, (w,), dtype)
        dense = self._mle_eval(rs_eq_ind(self._eq_tensor(z_vals), eq_r), query)
        succinct = eval_rs_eq(z_vals, query, eq_r)
        np.testing.assert_array_equal(_np_lanes(dense), _np_lanes(succinct))

    def test_point_length_mismatch_raises(self) -> None:
        dtype = zk_dtypes.binary_field_ghash
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            eval_rs_eq(
                _rand(1, (2,), dtype), _rand(2, (3,), dtype), _rand(3, (128,), dtype)
            )


class FieldBitWidthTest(absltest.TestCase):
    def test_sub_lane_field_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            field_bit_width(zk_dtypes.binary_field_t4)  # 16-bit


if __name__ == "__main__":
    absltest.main()
