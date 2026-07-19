# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


class _CollectRound(Round):
    """Halves a 1-element-per-factor carry; emits a heterogeneous dict message."""

    def __call__(self, state: Any, transcript: Any) -> Any:
        (xs,) = state
        half = xs.shape[-1] // 2
        msg = {"first": xs[0], "len": xs.shape[-1]}  # non-stackable on purpose
        return [xs[:half]], transcript, msg


class FoldRoundsTest(absltest.TestCase):
    """`fold_rounds` is the scheme-agnostic driver: it runs any `Round` N times and
    collects each round's (possibly heterogeneous, non-stackable) message. The
    multilinear sumcheck scan driver and its marker are tested in
    `zorch.sumcheck.testing`."""

    def test_collects_structured_messages_as_list(self) -> None:
        xs = fnp.arange(8, dtype=KB)
        _, _, msgs = fold_rounds(_CollectRound(), [xs], cheap_transcript(KB), 3)
        self.assertEqual([m["len"] for m in msgs], [8, 4, 2])
        self.assertEqual(len(msgs), 3)


if __name__ == "__main__":
    absltest.main()
