# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`sample_query_positions` draws in one squeeze; this pins it to the per-draw form.

WHIR's query convention fixes the draw ORDER and the CANONICAL low-limb reduction.
The batched draw is only legal because `Transcript.sample(count)` yields exactly
the sequence `count` single squeezes would, so the reference here IS the per-draw
loop — both the positions and the advanced transcript state must match, at counts
either side of the sponge rate (where a permutation fires mid-draw) and for every
base field, whose item size sets the bitcast's trailing shape.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array, lax, tree_util
from frx.typing import DTypeLike
from zk_dtypes import babybear_mont as BB
from zk_dtypes import koalabear_mont as KB

from zorch.pcs.whir._math import sample_query_positions
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

_RATE = 4
_STRIDE = 64


def _seeded(dtype: DTypeLike) -> DuplexTranscript:
    """A cheap transcript with something absorbed. A FRESH one is all zeros, and
    `CheapPermutation` maps zero to zero, so every squeeze would return 0 and every
    draw would be position 0 — an order- or reduction-breaking regression would
    slip through against that. Absorbing first makes the draws distinct."""
    return cheap_transcript(dtype=dtype, rate=_RATE).observe(
        fnp.arange(1, 6, dtype=fnp.uint32).astype(dtype)
    )


def _per_draw_reference(
    transcript: DuplexTranscript, stride: int, count: int, dtype: DTypeLike
) -> tuple[DuplexTranscript, Array]:
    """The form `sample_query_positions` replaced: one squeeze per index, each
    reduced on its canonical low limb."""
    if count == 0:
        return transcript, fnp.empty((0,), fnp.int32)
    positions = []
    t = transcript
    for _ in range(count):
        t, raw = t.sample(1)
        canonical = lax.bitcast_convert_type(raw, dtype).astype(fnp.uint32).reshape(-1)
        positions.append((canonical[0] % stride).astype(fnp.int32))
    return t, fnp.stack(positions)


class SampleQueryPositionsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # `_RATE` = squeezes per permutation: cross it in both directions so the
        # batched form's mid-draw permutation schedule is exercised.
        ("kb_none", KB, 0),
        ("kb_one", KB, 1),
        ("kb_below_rate", KB, _RATE - 1),
        ("kb_at_rate", KB, _RATE),
        ("kb_above_rate", KB, _RATE + 1),
        ("kb_many_rates", KB, 4 * _RATE + 3),
        ("bb_one", BB, 1),
        ("bb_at_rate", BB, _RATE),
        ("bb_many_rates", BB, 4 * _RATE + 3),
    )
    def test_batched_draw_matches_per_draw(self, dtype: DTypeLike, count: int) -> None:
        entry = _seeded(dtype)
        want_t, want = _per_draw_reference(entry, _STRIDE, count, dtype)
        got_t, got = sample_query_positions(entry, _STRIDE, count, dtype)

        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(want), err_msg="query positions"
        )
        self.assertEqual(got.shape, (count,))
        # Guard the comparison above from going vacuous: identical draws would
        # match any order or reduction.
        if count > 1:
            self.assertGreater(len(set(np.asarray(want).tolist())), 1)
        # The draw must leave the sponge where the per-draw form left it, or the
        # next challenge desynchronizes.
        for got_leaf, want_leaf in zip(
            tree_util.tree_leaves(got_t), tree_util.tree_leaves(want_t), strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(got_leaf), np.asarray(want_leaf), err_msg="transcript state"
            )

    def test_positions_are_in_range(self) -> None:
        _, positions = sample_query_positions(_seeded(KB), _STRIDE, 4 * _RATE, KB)
        self.assertTrue(bool(fnp.all(positions >= 0)))
        self.assertTrue(bool(fnp.all(positions < _STRIDE)))


if __name__ == "__main__":
    absltest.main()
