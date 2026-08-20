# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""A fused pair must be byte-identical to the two hops it replaces.

`_sample_scalar_after` puts a payload on the stream as part of a draw's framing
instead of as an absorb of its own, so the pair costs one marked region instead
of two. Its correctness is a claim about the stream — `absorb(a); absorb(b)` and
`absorb(a || b)` leave the same state — and a diverging byte here moves every
Fiat-Shamir draw after it. Both fused pairs are asserted against their unmerged
form, on both device rows.
"""
from absl.testing import absltest, parameterized

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax

from zorch.blake3_field_transcript import Blake3FieldTranscript
from zorch.sha256_field_transcript import Sha256FieldTranscript

# A hit at these difficulties lands far inside one window; `grind_search` tiles,
# so a narrow window returns the same nonce for a fraction of the hashing.
_TEST_WINDOW = 1024

# 0 takes the zero-witness short-circuit, 5 is the only value reaching
# `leading_zero_bits_ok`'s partial-byte branch, 9 reaches full >= 1.
_BITS = (0, 5, 9)

_ROWS = (
    ("blake3", lambda: Blake3FieldTranscript.new(b"fused", fnp.binary_field_ghash)),
    ("sha256", lambda: Sha256FieldTranscript.new(b"fused", fnp.binary_field_ghash)),
)


def _seeded(new):
    """A transcript with something absorbed, so the state is not the bare seed."""
    return new().observe(fnp.zeros(4, fnp.binary_field_ghash))


def _state(t):
    return [fnp.asarray(x).tobytes() for x in frx.tree_util.tree_leaves(t.state)]


class FusedPairTest(parameterized.TestCase):
    def _assert_same(self, unmerged, merged, what):
        t_a, r_a = unmerged
        t_b, r_b = merged
        self.assertEqual(
            fnp.asarray(r_a).tobytes(), fnp.asarray(r_b).tobytes(),
            f"challenge differs for {what}",
        )
        for i, (x, y) in enumerate(zip(_state(t_a), _state(t_b))):
            self.assertEqual(x, y, f"transcript state leaf {i} differs for {what}")

    @parameterized.named_parameters(*[
        (f"{row}_bits{b}", new, b) for row, new in _ROWS for b in _BITS
    ])
    def test_grind_and_sample_matches_unmerged(self, new, bits):
        t0 = _seeded(new)
        t_a, w_a = t0.grind(bits, chunk=_TEST_WINDOW)
        t_b, w_b, r_b = t0.grind_and_sample(bits, chunk=_TEST_WINDOW)
        self.assertEqual(int(w_a), int(w_b), f"witness differs at bits={bits}")
        self._assert_same(t_a.sample_scalar(), (t_b, r_b), f"bits={bits}")

    @parameterized.named_parameters(*[
        (f"{row}_n{n}", new, n) for row, new in _ROWS for n in (1, 3)
    ])
    def test_observe_scalar_and_sample_matches_unmerged(self, new, n):
        u = np.arange(2 * n, dtype=np.uint64).reshape(n, 2) + 7
        g = lax.bitcast_convert_type(fnp.asarray(u), fnp.binary_field_ghash)
        t0 = _seeded(new)
        self._assert_same(
            t0.observe_scalar(g).sample_scalar(),
            t0.observe_scalar_and_sample(g),
            f"n={n}",
        )


if __name__ == "__main__":
    absltest.main()
