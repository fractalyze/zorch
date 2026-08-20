# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`grind_and_sample` must be byte-identical to grind-then-sample_scalar.

The merge is what lets a round that grinds before it draws spend ONE marked
region instead of two, and its correctness is a claim about the stream:
`absorb(a); absorb(b)` and `absorb(a || b)` leave the same state, and a squeeze
zone absorbs its framing before reading. This asserts that on both wires rather
than trusting the argument -- a diverging byte here moves every Fiat-Shamir
draw after it.
"""
import unittest

import frx
import frx.numpy as fnp

from zorch.blake3_field_transcript import Blake3FieldTranscript
from zorch.sha256_field_transcript import Sha256FieldTranscript

_POW_BLOCK = 64  # flock pads its PoW pre-image to a whole block


def _leaves(t):
    return [fnp.asarray(x) for x in frx.tree_util.tree_leaves(t.state)]


class GrindAndSampleTest(unittest.TestCase):
    def _check(self, make, bits, chunk):
        t0 = make()
        # unmerged: the witness gets a marked region of its own
        t_a, w_a = t0.grind(bits, chunk=chunk)
        t_a, r_a = t_a.sample_scalar()
        # merged: the witness rides the draw's framing
        t_b, w_b, r_b = t0.grind_and_sample(bits, chunk=chunk)

        self.assertEqual(int(w_a), int(w_b), f"witness differs at bits={bits}")
        self.assertEqual(
            bytes(fnp.asarray(r_a).tobytes()),
            bytes(fnp.asarray(r_b).tobytes()),
            f"challenge differs at bits={bits}",
        )
        for i, (x, y) in enumerate(zip(_leaves(t_a), _leaves(t_b))):
            self.assertEqual(
                x.tobytes(), y.tobytes(),
                f"transcript state leaf {i} differs at bits={bits}",
            )

    def test_blake3_matches_unmerged(self):
        def make():
            t = Blake3FieldTranscript.new(
                b"grind-and-sample", fnp.binary_field_ghash,
                pow_preimage_bytes=_POW_BLOCK,
            )
            return t.observe(fnp.zeros(4, fnp.binary_field_ghash))

        for bits in (0, 1, 5, 9):
            with self.subTest(bits=bits):
                self._check(make, bits, 1 << 12)

    def test_sha256_matches_unmerged(self):
        def make():
            t = Sha256FieldTranscript.new(b"grind-and-sample", fnp.binary_field_ghash)
            return t.observe(fnp.zeros(4, fnp.binary_field_ghash))

        for bits in (0, 1, 5, 9):
            with self.subTest(bits=bits):
                self._check(make, bits, 1 << 12)

    def _check_observe(self, make, n):
        vals = fnp.arange(n, dtype=fnp.uint64).reshape(n, 1)
        vals = fnp.concatenate([vals, vals + 7], axis=1)
        g = frx.lax.bitcast_convert_type(vals, fnp.binary_field_ghash)
        t0 = make()
        t_a = t0.observe_scalar(g)
        t_a, r_a = t_a.sample_scalar()
        t_b, r_b = t0.observe_scalar_and_sample(g)
        self.assertEqual(
            bytes(fnp.asarray(r_a).tobytes()), bytes(fnp.asarray(r_b).tobytes()),
            f"challenge differs at n={n}",
        )
        for i, (x, y) in enumerate(zip(_leaves(t_a), _leaves(t_b))):
            self.assertEqual(
                x.tobytes(), y.tobytes(),
                f"transcript state leaf {i} differs at n={n}",
            )

    def test_blake3_observe_and_sample_matches_unmerged(self):
        def make():
            t = Blake3FieldTranscript.new(
                b"grind-and-sample", fnp.binary_field_ghash,
                pow_preimage_bytes=_POW_BLOCK,
            )
            return t.observe(fnp.zeros(4, fnp.binary_field_ghash))

        for n in (1, 3):
            with self.subTest(n=n):
                self._check_observe(make, n)

    def test_sha256_observe_and_sample_matches_unmerged(self):
        def make():
            t = Sha256FieldTranscript.new(b"grind-and-sample", fnp.binary_field_ghash)
            return t.observe(fnp.zeros(4, fnp.binary_field_ghash))

        for n in (1, 3):
            with self.subTest(n=n):
                self._check_observe(make, n)

    def test_window_does_not_change_the_wire(self):
        """`grind_search` tiles ascending, so a narrower window still returns the
        lowest hit -- the merged path must inherit that."""
        def make():
            t = Blake3FieldTranscript.new(
                b"grind-and-sample", fnp.binary_field_ghash,
                pow_preimage_bytes=_POW_BLOCK,
            )
            return t.observe(fnp.zeros(4, fnp.binary_field_ghash))

        wide = make().grind_and_sample(9, chunk=1 << 14)
        narrow = make().grind_and_sample(9, chunk=1 << 8)
        self.assertEqual(int(wide[1]), int(narrow[1]))
        self.assertEqual(
            bytes(fnp.asarray(wide[2]).tobytes()),
            bytes(fnp.asarray(narrow[2]).tobytes()),
        )


if __name__ == "__main__":
    unittest.main()
