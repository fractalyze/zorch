# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The traced-schedule transition path, and its capacity guard.

`jagged_layer_transition` takes its fold policy either as a bare host sequence
or as `(traced_counts, out_width)`. Only the host form had coverage, and the
traced form is the one a real consumer uses -- sp1-zorch derives its per-layer
widths from a capacity class and hands them down, because a width that tracks an
input's row counts would key the round zone per shard.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.circuit import jagged_layer_transition
from zorch.logup_gkr.testing import (
    host_counts,
    jagged_fold_schedules,
    random_jagged_layer,
)

_KB = zk_dtypes.koalabear_mont
_RC = (12, 7, 20, 9)


def _planes(layer):
    return tuple(
        np.asarray(a).tobytes()
        for a in (
            layer.numerator_0,
            layer.numerator_1,
            layer.denominator_0,
            layer.denominator_1,
        )
    )


class TracedScheduleTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.layer = random_jagged_layer(5, _RC)
        self.schedule = jagged_fold_schedules(self.layer)[0]

    def test_traced_and_host_schedules_agree(self) -> None:
        """Same policy through both spellings must fold to the same layer --
        the traced counts are an operand, so nothing about the fold may depend
        on their being host-readable."""
        host = jagged_layer_transition(self.layer, self.schedule)
        traced = jagged_layer_transition(
            self.layer, fnp.asarray(self.schedule, fnp.int32), sum(self.schedule)
        )

        self.assertEqual(host.width, traced.width)
        self.assertEqual(_planes(host), _planes(traced))
        self.assertEqual(host_counts(host), host_counts(traced))

    def test_traced_schedule_honours_a_wider_declared_capacity(self) -> None:
        """A consumer declaring more than the live size gets it: the slack is
        dead region, so the live prefix must match the zero-slack fold."""
        slack = sum(self.schedule) + 8
        wide = jagged_layer_transition(
            self.layer, fnp.asarray(self.schedule, fnp.int32), slack
        )
        tight = jagged_layer_transition(self.layer, self.schedule)

        self.assertEqual(wide.width, slack)
        live = sum(self.schedule)
        for wide_plane, tight_plane in zip(
            (wide.numerator_0, wide.denominator_0),
            (tight.numerator_0, tight.denominator_0),
            strict=True,
        ):
            np.testing.assert_array_equal(
                np.asarray(wide_plane)[:live], np.asarray(tight_plane)[:live]
            )

    def test_a_traced_schedule_needs_an_explicit_capacity(self) -> None:
        """Traced counts are unreadable host-side, so there is nothing to
        default the width to."""
        with self.assertRaisesRegex(ValueError, "explicit out_width"):
            jagged_layer_transition(
                self.layer, fnp.asarray(self.schedule, fnp.int32)
            )

    def test_a_capacity_too_narrow_for_a_host_schedule_is_rejected(self) -> None:
        """The one truncation a host guard can see. Where the counts are traced
        this stays the consumer's obligation, but a host schedule states its own
        live size, and a too-narrow capacity otherwise surfaces much later as a
        dead-region read.
        """
        with self.assertRaisesRegex(ValueError, "cannot hold the schedule"):
            jagged_layer_transition(self.layer, self.schedule, sum(self.schedule) - 1)

    def test_an_exact_capacity_is_accepted(self) -> None:
        exact = jagged_layer_transition(
            self.layer, self.schedule, sum(self.schedule)
        )
        self.assertEqual(exact.width, sum(self.schedule))


if __name__ == "__main__":
    absltest.main()
