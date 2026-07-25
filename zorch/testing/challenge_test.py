# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Challenge-policy field selection and squeeze-stream equivalences."""

from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.challenge import ChallengePolicy, challenge_limbs
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
    def test_transcript_field_policy_is_the_raw_squeeze_stream(self) -> None:
        # Naming the transcript's own field reinterprets one squeeze as itself,
        # so it must consume exactly the squeezes a bare `sample` would.
        policy, a = ChallengePolicy(KB).sample_many(cheap_transcript(KB), 3)
        raw, b = cheap_transcript(KB).sample(3)
        self.assertTrue(bool(fnp.all(a == b)))
        _, after_policy = policy.sample(1)
        _, after_raw = raw.sample(1)
        self.assertTrue(bool(fnp.all(after_policy == after_raw)))

    def test_extension_policy_promotes_a_base_field_transcript(self) -> None:
        # The reason the field is named rather than read off a running value:
        # an extension policy draws extension challenges from a base-field
        # transcript, which raises the soundness floor of a base-field claim.
        _, value = ChallengePolicy(KBX4).sample(cheap_transcript(KB))
        self.assertEqual(value.dtype, KBX4)

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


if __name__ == "__main__":
    absltest.main()
