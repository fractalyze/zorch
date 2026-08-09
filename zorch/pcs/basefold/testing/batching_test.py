# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Staggered partial-Lagrange batching: weight allocation and the degenerate
single-column case."""
from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.basefold.batching import (
    batch_staggered,
    partial_lagrange,
    sample_staggered_coeffs,
)
from zorch.testkit.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_ext_field
from zorch.transcript import DuplexTranscript


def _rand(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    from zk_dtypes import koalabear_mont as F

    return rand_ext_field(seed, shape, F, EF)


class BatchingTest(absltest.TestCase):
    def test_partial_lagrange_one_var_is_eq_basis(self) -> None:
        r = _rand(1, ())
        coeffs = partial_lagrange(fnp.reshape(r, (1,)))
        one = fnp.ones((), EF)
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
        self.assertEqual(coeffs.tolist(), fnp.ones(1, EF).tolist())
        # No squeeze for width 1: the same transcript is returned untouched.
        self.assertIs(t2, t)

    def test_sample_staggered_covers_total_width(self) -> None:
        t = DuplexTranscript.new(koalabear16_perm(), rate=8)
        _, coeffs = sample_staggered_coeffs(t, 3, EF)
        # log2_ceil(3) = 2 -> 2^2 = 4 weights, enough to stagger 3 columns.
        self.assertGreaterEqual(coeffs.shape[0], 3)

    def test_partial_lagrange_is_msb_first(self) -> None:
        # Pin the native index orientation: point[0] is the MSB of the table
        # index — table = [(1-r0)(1-r1), (1-r0)r1, r0(1-r1), r0·r1].
        r = _rand(8, (2,))
        one = fnp.ones((), EF)
        r0, r1 = r[0], r[1]
        table = partial_lagrange(r)
        expected = [
            (one - r0) * (one - r1),
            (one - r0) * r1,
            r0 * (one - r1),
            r0 * r1,
        ]
        self.assertEqual(table.tolist(), [e.tolist() for e in expected])

    def test_sample_staggered_lsb_first_bit_reverses_the_table(self) -> None:
        # Both orientations consume the transcript identically; the tables are
        # index-bit-reversals of each other: lsb[i] == msb[bitrev(i)].
        t = DuplexTranscript.new(koalabear16_perm(), rate=8)
        t_msb, msb = sample_staggered_coeffs(t, 8, EF)
        t_lsb, lsb = sample_staggered_coeffs(t, 8, EF, lsb_first=True)
        bitrev3 = [0, 4, 2, 6, 1, 5, 3, 7]
        self.assertEqual(lsb.tolist(), msb[fnp.array(bitrev3)].tolist())
        # Same post-state: a follow-up squeeze from either transcript matches.
        _, a = t_msb.sample()
        _, b = t_lsb.sample()
        self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    absltest.main()
