# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Challenge-policy field selection and squeeze-stream equivalences."""

from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.challenge import DEFAULT_CHALLENGES, ChallengePolicy, challenge_limbs
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont
KBX4 = zk_dtypes.koalabearx4_mont


class ChallengeLimbsTest(absltest.TestCase):
    def test_extension_takes_one_squeeze_per_degree(self) -> None:
        self.assertEqual(challenge_limbs(KBX4), 4)

    def test_base_field_takes_one_squeeze(self) -> None:
        self.assertEqual(challenge_limbs(KB), 1)


class ChallengeStreamTest(absltest.TestCase):
    def test_naming_the_transcript_field_matches_the_native_default(self) -> None:
        # An explicit base-field target reinterprets one squeeze as itself, so it
        # must not consume a different number of squeezes than the default.
        native, a = DEFAULT_CHALLENGES.sample_many(cheap_transcript(KB), 3)
        explicit, b = ChallengePolicy(KB).sample_many(cheap_transcript(KB), 3)
        self.assertTrue(bool(fnp.all(a == b)))
        _, after_native = native.sample(1)
        _, after_explicit = explicit.sample(1)
        self.assertTrue(bool(fnp.all(after_native == after_explicit)))

    def test_batched_sampling_matches_repeated_single_sampling(self) -> None:
        # `sample_many` squeezes every limb in one call; that must produce the
        # same stream as sampling the challenges one at a time.
        policy = ChallengePolicy(KBX4)
        batched_t, batched = policy.sample_many(cheap_transcript(KB), 3)
        one_at_a_time = []
        loop_t: Transcript = cheap_transcript(KB)
        for _ in range(3):
            loop_t, value = policy.sample(loop_t)
            one_at_a_time.append(value)
        self.assertTrue(bool(fnp.all(batched == fnp.stack(one_at_a_time))))
        _, after_batched = batched_t.sample(1)
        _, after_loop = loop_t.sample(1)
        self.assertTrue(bool(fnp.all(after_batched == after_loop)))

    def test_observe_and_sample_matches_separate_observe_then_sample(self) -> None:
        # The fused hop is an optimization, never a different transcript.
        policy = ChallengePolicy(KBX4)
        values = fnp.arange(4, dtype=KB)
        fused_t, fused = policy.observe_and_sample(cheap_transcript(KB), values)
        split_t, split = policy.sample(cheap_transcript(KB).observe(values))
        self.assertTrue(bool(fnp.all(fused == split)))
        _, after_fused = fused_t.sample(1)
        _, after_split = split_t.sample(1)
        self.assertTrue(bool(fnp.all(after_fused == after_split)))


class LimbOnlyPolicyTest(absltest.TestCase):
    def test_limb_only_policy_needs_a_value_dtype(self) -> None:
        policy = ChallengePolicy(limbs=4)
        self.assertTrue(policy.needs_value_dtype)
        self.assertFalse(ChallengePolicy(KBX4).needs_value_dtype)
        self.assertFalse(DEFAULT_CHALLENGES.needs_value_dtype)

    def test_limb_only_policy_takes_the_supplied_value_field(self) -> None:
        policy = ChallengePolicy(limbs=4)
        _, value = policy.sample(cheap_transcript(KB), KBX4)
        self.assertEqual(value.dtype, KBX4)

    def test_limb_only_policy_without_a_value_dtype_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChallengePolicy(limbs=4).sample(cheap_transcript(KB))


if __name__ == "__main__":
    absltest.main()
