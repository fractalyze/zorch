# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`BasefoldChoreography` native defaults: each hook's real behavior against a
`cheap_transcript` (a genuine, fast `DuplexTranscript`) — no consumer wired in
yet, so these pin the wire Tasks 3-4 must reproduce byte-for-byte."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldConfig
from zorch.pcs.fold import sample_positions
from zorch.testkit.transcript import cheap_transcript


def _states_equal(a: object, b: object) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        bool(jnp.array_equal(x, y)) for x, y in zip(la, lb, strict=True)
    )


class RoundMessageTest(absltest.TestCase):
    def test_stacks_zero_and_one(self) -> None:
        chor = BasefoldChoreography()
        zero_val, one_val = F(3), F(5)
        got = chor.round_message(zero_val, one_val)
        self.assertTrue(bool(jnp.array_equal(got, jnp.stack([zero_val, one_val]))))


class ReduceClaimTest(absltest.TestCase):
    def test_additive_combine(self) -> None:
        # native: s(0) + r*s(1) -- NOT the affine (1-r)*s(0) + r*s(1) bind.
        chor = BasefoldChoreography()
        msg = jnp.stack([F(3), F(5)])
        r = F(2)
        got = chor.reduce_claim(F(0), msg, r)
        self.assertEqual(got.tolist(), (F(3) + r * F(5)).tolist())

    def test_ignores_running_claim(self) -> None:
        chor = BasefoldChoreography()
        msg = jnp.stack([F(3), F(5)])
        r = F(2)
        a = chor.reduce_claim(F(0), msg, r)
        b = chor.reduce_claim(F(99), msg, r)
        self.assertEqual(a.tolist(), b.tolist())


class ObserveFinalTest(absltest.TestCase):
    def test_observes_whole_final_poly(self) -> None:
        chor = BasefoldChoreography()
        final_poly = jnp.arange(4, dtype=F)
        got = chor.observe_final(cheap_transcript(F), final_poly)
        want = cheap_transcript(F).observe(final_poly)
        self.assertTrue(_states_equal(got, want))


class GrindScheduleTest(absltest.TestCase):
    def test_fold_grind_bits_default_none(self) -> None:
        chor = BasefoldChoreography()
        self.assertIsNone(chor.fold_grind_bits(0, 0))
        self.assertIsNone(chor.fold_grind_bits(3, 1))

    def test_query_grind_bits_default_none(self) -> None:
        chor = BasefoldChoreography()
        self.assertIsNone(chor.query_grind_bits(0))
        self.assertIsNone(chor.query_grind_bits(2))

    def test_num_pow_witnesses_default_zero(self) -> None:
        chor = BasefoldChoreography()
        config = BasefoldConfig(num_vars=4, num_queries=4)
        self.assertEqual(chor.num_pow_witnesses(config), 0)

    def test_num_pow_witnesses_counts_a_scheduled_grind(self) -> None:
        @dataclass(frozen=True)
        class _Grinding(BasefoldChoreography):
            def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
                del fold_idx
                return 8 if level < 2 else None

            def query_grind_bits(self, level: int) -> int | None:
                del level
                return 12

        chor = _Grinding()
        config = BasefoldConfig(num_vars=4, num_queries=4)
        # 2 grinding fold rounds (level 0, 1) + the one query-phase grind.
        self.assertEqual(chor.num_pow_witnesses(config), 3)


class GrindRoundTripTest(absltest.TestCase):
    def test_grind_and_check_grind_agree(self) -> None:
        chor = BasefoldChoreography()
        prover_t, witness = chor.grind(cheap_transcript(F), 4)
        verifier_t, ok = chor.check_grind(cheap_transcript(F), 4, witness)
        self.assertTrue(bool(ok))
        self.assertTrue(_states_equal(prover_t, verifier_t))


class SharedHookDefaultsTest(absltest.TestCase):
    def test_eager_messages_default_false(self) -> None:
        self.assertFalse(BasefoldChoreography().eager_messages)

    def test_observe_root_default(self) -> None:
        chor = BasefoldChoreography()
        root = jnp.arange(8, dtype=F)
        got = chor.observe_root(cheap_transcript(F), root)
        want = cheap_transcript(F).observe(root)
        self.assertTrue(_states_equal(got, want))

    def test_observe_message_default(self) -> None:
        chor = BasefoldChoreography()
        msg = jnp.stack([F(1), F(2)])
        got = chor.observe_message(cheap_transcript(F), msg)
        want = cheap_transcript(F).observe(msg)
        self.assertTrue(_states_equal(got, want))

    def test_bind_statement_observes_root_point_value_in_order(self) -> None:
        chor = BasefoldChoreography()
        root, point, value = F(1), F(2), F(3)
        got = chor.bind_statement(cheap_transcript(F), root, point, value)
        want = cheap_transcript(F).observe(root).observe(point).observe(value)
        self.assertTrue(_states_equal(got, want))

    def test_bind_statement_raises_without_point(self) -> None:
        chor = BasefoldChoreography()
        with self.assertRaisesRegex(ValueError, "override bind_statement"):
            chor.bind_statement(cheap_transcript(F), F(1), None, F(3))

    def test_fold_challenge_default_fuses_observe_and_sample(self) -> None:
        chor = BasefoldChoreography()
        msg = jnp.stack([F(1), F(2)])
        got_t, got_r = chor.fold_challenge(cheap_transcript(F), msg, 0, 0)
        want_t, want_r = cheap_transcript(F).observe_and_sample(msg, 1)
        self.assertTrue(_states_equal(got_t, want_t))
        self.assertEqual(got_r.tolist(), want_r[0].tolist())

    def test_fold_challenge_rejects_none_msg_under_lazy_default(self) -> None:
        chor = BasefoldChoreography()
        with self.assertRaisesRegex(ValueError, "eager choreography"):
            chor.fold_challenge(cheap_transcript(F), None, 0, 0)

    def test_sample_queries_delegates_to_sample_positions(self) -> None:
        chor = BasefoldChoreography()
        got_t, got_positions = chor.sample_queries(cheap_transcript(F), 16, 3)
        want_t, want_positions = sample_positions(cheap_transcript(F), 16, 3)
        self.assertTrue(_states_equal(got_t, want_t))
        self.assertEqual(got_positions.tolist(), want_positions.tolist())
        self.assertTrue(bool(jnp.all(got_positions >= 0)))
        self.assertTrue(bool(jnp.all(got_positions < 16)))


if __name__ == "__main__":
    absltest.main()
