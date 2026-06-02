# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reed-Solomon encode: checked against independent polynomial evaluation.

The oracle never reuses the encoder. The NTT evaluation domain is recovered
straight from `lax.fft` of an impulse (NTT(e_1)_j = w^j), then the codeword is
compared to a Horner evaluation of the message polynomial on that domain — a
path that shares no code with pad + NTT.
"""

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import lax

from zorch.coding import LinearCode, ReedSolomon
from zorch.testkit.random_field import rand_field

F = zk_dtypes.koalabear


def _domain(n, dtype):
    """The order-n NTT evaluation domain [w^0, ..., w^{n-1}] for the canonical
    root, recovered independently of the encoder: NTT(e_1)_j = w^j."""
    e1 = jnp.zeros((n,), dtype).at[1].set(jnp.ones((), dtype))
    return lax.fft(e1, "FFT", n)


def _horner(coeffs, points):
    """Evaluate the polynomial with `coeffs` at every point in `points`."""
    acc = points * jnp.zeros((), points.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * points + coeffs[i]
    return acc


class ReedSolomonTest(absltest.TestCase):
    def test_implements_linear_code_protocol(self):
        rs = ReedSolomon(message_len=4, blowup=2, dtype=F)
        self.assertIsInstance(rs, LinearCode)
        self.assertEqual(rs.message_len, 4)
        self.assertEqual(rs.block_len, 8)
        self.assertEqual(rs.dtype, F)

    def test_encode_matches_polynomial_evaluation(self):
        k, blowup = 4, 4
        rs = ReedSolomon(k, blowup, F)
        coeffs = rand_field(1, (k,), F)
        want = _horner(coeffs, _domain(k * blowup, F))
        self.assertTrue(bool(jnp.all(rs.encode(coeffs) == want)))

    def test_codeword_is_low_degree(self):
        k, blowup = 8, 2
        rs = ReedSolomon(k, blowup, F)
        coeffs = rand_field(2, (k,), F)
        rec = lax.fft(rs.encode(coeffs), "IFFT", k * blowup)
        self.assertTrue(bool(jnp.all(rec[:k] == coeffs)))
        self.assertTrue(bool(jnp.all(rec[k:] == jnp.zeros(k * blowup - k, F))))

    def test_encode_is_linear(self):
        rs = ReedSolomon(4, 2, F)
        a, b = rand_field(3, (4,), F), rand_field(4, (4,), F)
        self.assertTrue(bool(jnp.all(rs.encode(a + b) == rs.encode(a) + rs.encode(b))))

    def test_encode_batched_rows(self):
        k, blowup, rows = 4, 2, 3
        rs = ReedSolomon(k, blowup, F)
        msg = rand_field(5, (rows, k), F)
        cw = rs.encode(msg)
        self.assertEqual(cw.shape, (rows, k * blowup))
        for r in range(rows):
            self.assertTrue(bool(jnp.all(cw[r] == rs.encode(msg[r]))))

    def test_coset_encode_matches_shifted_evaluation(self):
        k, blowup = 4, 2
        shift = jnp.array(3, dtype=F)
        rs = ReedSolomon(k, blowup, F, coset_shift=shift)
        coeffs = rand_field(6, (k,), F)
        want = _horner(coeffs, shift * _domain(k * blowup, F))
        self.assertTrue(bool(jnp.all(rs.encode(coeffs) == want)))

    def test_wrong_message_length_raises(self):
        rs = ReedSolomon(4, 2, F)
        with self.assertRaises(ValueError):
            rs.encode(rand_field(7, (5,), F))

    def test_non_power_of_two_raises(self):
        with self.assertRaises(ValueError):
            ReedSolomon(message_len=3, blowup=2, dtype=F)
        with self.assertRaises(ValueError):
            ReedSolomon(message_len=4, blowup=3, dtype=F)


if __name__ == "__main__":
    absltest.main()
