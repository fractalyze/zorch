# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-layer element ladder: same proof, one dispatch per layer less.

Under a flat cap every proved layer is laid into a round-0-wide buffer by a
separate `_lay_prefix_many` dispatch before its round zone runs. A ladder makes
each layer arrive at the width its own zone runs, so the lay-in degenerates to
a passthrough. That is only worth anything if the proof does not move and the
class stays shard-invariant, which is what these pin.
"""

from __future__ import annotations

import random

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest

from zorch import transcript as T
from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import (
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import _jagged_round_zone
from zorch.logup_gkr.jagged_stage import (
    JaggedGkrWitness,
    JaggedLogUpGkrProver,
    _caps_from_widths,
)
from zorch.logup_gkr.stage import LogUpOutputClaim
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    caps_for,
    element_ladder_for,
    host_counts,
    jagged_fold_schedules,
    random_jagged_layer,
)
from zorch.sumcheck.jagged import buffers as BUF
from zorch.testkit.koalabear16 import koalabear16_perm

_KB = zk_dtypes.koalabear_mont
_CH = ChallengePolicy(_KB)
_RC = (1024, 700, 1500, 900)


def _leaves(root):
    seen, found = set(), []

    def walk(o, depth=0):
        if depth > 8 or id(o) in seen:
            return
        seen.add(id(o))
        if hasattr(o, "is_ready") and hasattr(o, "shape"):
            found.append(o)
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x, depth + 1)
            return
        for name in getattr(o, "__dataclass_fields__", {}) or {}:
            walk(getattr(o, name, None), depth + 1)

    walk(root)
    return found


def _host_schedules(counts):
    """Host mirror of `jagged_fold_schedules`, so the domination check sweeps
    layouts without building a layer per candidate."""
    out = []
    cur = tuple(counts)
    while max(cur) > 1:
        folded = tuple((rc + 1) // 2 for rc in cur)
        cur = tuple(fc if fc == 1 else fc + fc % 2 for fc in folded)
        out.append(cur)
    return out


class ElementLadderTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.first = random_jagged_layer(7, _RC)
        self.scheds = jagged_fold_schedules(self.first)
        floor = build_jagged_pyramid(self.first)[-1]
        self.claim = LogUpOutputClaim(
            extract_jagged_outputs(floor), len(self.scheds)
        )
        self.caps = caps_for(host_counts(self.first), len(self.scheds))
        self.witness = JaggedGkrWitness(self.first, self.scheds)
        self.ladder = element_ladder_for(
            self.caps, self.first.num_batches, len(self.scheds)
        )

    def _prove(self, prover):
        transcript = T.DuplexTranscript.new(koalabear16_perm(), rate=8)
        out = prover.prove(self.claim, self.witness, transcript)
        arrays = _leaves(out)
        frx.block_until_ready(arrays)
        # Field dtypes carry no Python buffer protocol, so the bytes come off
        # a numpy view rather than `memoryview`.
        return [np.asarray(a).tobytes() for a in arrays]

    def test_ladder_narrows_every_layer_below_the_flat_cap(self) -> None:
        self.assertEqual(self.ladder[0], self.caps.elements)
        self.assertLess(self.ladder[-1], self.ladder[0])
        for wide, narrow in zip(self.ladder, self.ladder[1:], strict=False):
            self.assertLessEqual(narrow, wide)

    def test_proof_is_byte_identical_to_the_flat_cap_path(self) -> None:
        """The ladder only changes where dead padding sits; every round masks
        its reads by the live prefix, so the transcript and the proof must not
        move by a byte."""
        flat = self._prove(JaggedLogUpGkrProver(self.caps, _CH))
        laddered = self._prove(
            JaggedLogUpGkrProver(self.caps, _CH, element_ladder=self.ladder)
        )
        self.assertEqual(flat, laddered)

    def test_ladder_removes_the_per_layer_lay_in_dispatch(self) -> None:
        """The point of the ladder: layers arrive at their zone's own width,
        so the pool never has to lay one in."""
        calls = []
        original = BUF._lay_prefix_many
        BUF._lay_prefix_many = lambda dsts, srcs: (
            calls.append(1),
            original(dsts, srcs),
        )[1]
        try:
            self._prove(JaggedLogUpGkrProver(self.caps, _CH))
            flat_calls = len(calls)
            calls.clear()
            self._prove(
                JaggedLogUpGkrProver(self.caps, _CH, element_ladder=self.ladder)
            )
            ladder_calls = len(calls)
        finally:
            BUF._lay_prefix_many = original

        self.assertGreater(flat_calls, 0)
        self.assertEqual(ladder_calls, 0)

    def test_ladder_keeps_the_class_shard_invariant(self) -> None:
        """A ladder derived from row counts would recompile per shard -- the
        zone takes it as a static arg. Proving a different input of the same
        class must add no executables.
        """
        prover = JaggedLogUpGkrProver(self.caps, _CH, element_ladder=self.ladder)
        self._prove(prover)
        after_first = _jagged_round_zone._cache_size()

        other = random_jagged_layer(11, (980, 664, 1422, 848))
        scheds = jagged_fold_schedules(other)
        self.assertEqual(len(scheds), len(self.scheds))
        floor = build_jagged_pyramid(other)[-1]
        transcript = T.DuplexTranscript.new(koalabear16_perm(), rate=8)
        frx.block_until_ready(
            _leaves(
                prover.prove(
                    LogUpOutputClaim(extract_jagged_outputs(floor), len(scheds)),
                    JaggedGkrWitness(other, scheds),
                    transcript,
                )
            )
        )

        self.assertEqual(_jagged_round_zone._cache_size(), after_first)

    def test_ladder_dominates_the_widths_its_schedule_policy_produces(self) -> None:
        """Across many layouts, not just this suite's.

        `jagged_fold_schedules` rounds each folded count up to even, so a
        bound that only halves under-counts whenever that rounding fires. It
        does not fire for `_RC` -- every halving there is already even -- so a
        single hand-picked layout cannot see the gap; ~15% of random ones can.
        """
        rng = random.Random(0)
        for _ in range(200):
            counts = tuple(
                rng.randint(1, 2000) for _ in range(rng.choice([2, 4, 8]))
            )
            scheds = _host_schedules(counts)
            caps = caps_for(counts, len(scheds))
            ladder = element_ladder_for(caps, len(counts), len(scheds))
            widths = [caps.elements] + [sum(s) for s in scheds]
            for k, cap in enumerate(ladder):
                self.assertLessEqual(
                    widths[k], cap, f"layout {counts}, layer {k} exceeds its cap"
                )

    def test_a_too_narrow_capacity_is_rejected_host_side(self) -> None:
        """The guard that would have caught the bug above at its source."""
        layer = random_jagged_layer(3, (8, 6))
        with self.assertRaisesRegex(ValueError, "cannot hold the schedule"):
            jagged_layer_transition(layer, (4, 3), 4)

    def test_a_traced_schedule_declaring_its_widths_needs_no_ladder(self) -> None:
        """The production shape: counts ride as a device array and the
        consumer declares each capacity itself. Those declared widths are
        already the ladder, so a plain prover must pick them up -- passing
        `element_ladder=` on top would be restating them.
        """
        traced = [
            (fnp.asarray(counts, fnp.int32), width)
            for counts, width in zip(
                self.scheds, [*self.ladder[1:], sum(self.scheds[-1])], strict=True
            )
        ]
        witness = JaggedGkrWitness(self.first, traced)

        calls = []
        original = BUF._lay_prefix_many
        BUF._lay_prefix_many = lambda d, s: (calls.append(1), original(d, s))[1]
        try:
            transcript = T.DuplexTranscript.new(koalabear16_perm(), rate=8)
            out = JaggedLogUpGkrProver(self.caps, _CH).prove(
                self.claim, witness, transcript
            )
            arrays = _leaves(out)
            frx.block_until_ready(arrays)
            proof = [np.asarray(a).tobytes() for a in arrays]
        finally:
            BUF._lay_prefix_many = original

        self.assertEqual(calls, [])
        self.assertEqual(proof, self._prove(JaggedLogUpGkrProver(self.caps, _CH)))

    def test_default_declines_widths_the_rounds_cannot_take(self) -> None:
        """The natural fold widths land on 5, 10, 18 near the floor. Rounding
        those up would set the cap above the width the layer was built at,
        which hands the lay-in straight back -- so the default must decline,
        not round.
        """
        self.assertIsNone(_caps_from_widths([4124, 2062, 5], 4124))
        self.assertIsNone(_caps_from_widths([8192, 4096], 4124))
        self.assertIsNone(_caps_from_widths([1024, 2048], 4124))
        self.assertEqual(
            _caps_from_widths([4124, 2064, 1036], 4124), (4124, 2064, 1036)
        )

    def test_ladder_length_is_checked_against_the_depth(self) -> None:
        prover = JaggedLogUpGkrProver(self.caps, _CH, element_ladder=self.ladder[:-1])
        with self.assertRaisesRegex(ValueError, "one cap per proved layer"):
            self._prove(prover)


if __name__ == "__main__":
    absltest.main()
