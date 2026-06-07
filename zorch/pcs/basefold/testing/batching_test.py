# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Staggered partial-Lagrange batching: weight allocation and the degenerate
single-column case."""
from __future__ import annotations

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabearx4_mont as EF

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.batching import (
    batch_staggered,
    partial_lagrange,
    sample_staggered_coeffs,
)
from zorch.testkit.random_field import rand_ext_field
from zorch.transcript import DuplexTranscript


def _rand(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    from zk_dtypes import koalabear_mont as F

    return rand_ext_field(seed, shape, F, EF)


class BatchingTest(absltest.TestCase):
    def test_partial_lagrange_one_var_is_eq_basis(self) -> None:
        r = _rand(1, ())
        coeffs = partial_lagrange(jnp.reshape(r, (1,)))
        one = jnp.ones((), EF)
        # eq(b, r) over b in {0,1} = {1-r, r} in some order; both must appear and
        # sum to 1.
        self.assertEqual(coeffs.shape, (2,))
        self.assertEqual(coeffs.sum().tolist(), one.tolist())

    def test_batch_staggered_allocates_per_round_widths(self) -> None:
        # Two rounds of widths [2, 1]; round 0 takes coeffs[0:2], round 1 coeffs[2].
        c = _rand(2, (4,))
        r0 = _rand(3, (5, 2))  # [rows, w0]
        r1 = _rand(4, (5, 1))  # [rows, w1]
        out = batch_staggered([r0, r1], c)
        expected = r0[:, 0] * c[0] + r0[:, 1] * c[1] + r1[:, 0] * c[2]
        self.assertEqual(out.shape, (5,))
        self.assertEqual(out.tolist(), expected.tolist())

    def test_batch_staggered_scalar_claims(self) -> None:
        # Claims are 1-D per round; the leading axis is empty -> scalar result.
        c = _rand(5, (4,))
        r0 = _rand(6, (2,)).reshape(1, 2)
        r1 = _rand(7, (1,)).reshape(1, 1)
        out = batch_staggered([r0[0].reshape(2), r1[0].reshape(1)], c)
        # 1-D inputs collapse the leading axis: result is a scalar.
        self.assertEqual(out.shape, ())

    def test_sample_staggered_width1_is_unit(self) -> None:
        t = DuplexTranscript.new(koalabear16_perm(), rate=8)
        t2, coeffs = sample_staggered_coeffs(t, 1, EF)
        self.assertEqual(coeffs.tolist(), jnp.ones(1, EF).tolist())
        # No squeeze for width 1: the same transcript is returned untouched.
        self.assertIs(t2, t)

    def test_sample_staggered_covers_total_width(self) -> None:
        t = DuplexTranscript.new(koalabear16_perm(), rate=8)
        _, coeffs = sample_staggered_coeffs(t, 3, EF)
        # log2_ceil(3) = 2 -> 2^2 = 4 weights, enough to stagger 3 columns.
        self.assertGreaterEqual(coeffs.shape[0], 3)


if __name__ == "__main__":
    absltest.main()
