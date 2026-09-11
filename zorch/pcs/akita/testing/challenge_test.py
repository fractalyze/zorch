# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The challenge-policy seam — the byte-count/parser pairing, the sparse
instance's shape, and that a second, unrelated policy satisfies the same
protocol.

The last one is the point of the seam existing: `zorch/lnp`'s σ₋₁-invariant
challenge set knows nothing about this package, so if it conforms, the
protocol describes challenge sets rather than one sampler.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from lattice_frx.sampler import fixed_weight_ternary_bytes_needed

from zorch.byte_transcript import ByteHashTranscript, ByteTranscript
from zorch.lnp.challenge import ChallengeParams
from zorch.pcs.akita.challenge import (
    ChallengePolicy,
    FixedWeightTernary,
    squeeze_challenge,
)
from zorch.testkit.byte_hash import HostSha256

_D = 64
_WEIGHT = 8


def _stream(nbytes: int) -> bytes:
    """A fixed byte block of exactly the length the policy quotes — the
    sampler consumes its whole stream and refuses any other length."""
    return bytes(range(256)) * (nbytes // 256) + bytes(range(nbytes % 256))


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return ByteHashTranscript.new(b"akita-challenge-test", HostSha256()).observe_bytes(
        tag
    )


class FixedWeightTernaryTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.policy = FixedWeightTernary(_D, _WEIGHT)

    def test_byte_count_matches_the_sampler_companion(self) -> None:
        self.assertEqual(
            self.policy.bytes_needed,
            fixed_weight_ternary_bytes_needed(_WEIGHT, _D, self.policy.fail_prob),
        )

    def test_draw_is_sparse_and_ternary(self) -> None:
        challenge = self.policy.from_bytes(_stream(self.policy.bytes_needed))
        self.assertLen(challenge, _D)
        self.assertEqual(int(np.count_nonzero(challenge)), _WEIGHT)
        self.assertEqual(set(np.unique(challenge)) - {0}, {-1, 1})

    def test_identical_bytes_give_an_identical_challenge(self) -> None:
        data = _stream(self.policy.bytes_needed)
        np.testing.assert_array_equal(
            self.policy.from_bytes(data), self.policy.from_bytes(data)
        )

    def test_refuses_a_weight_the_degree_cannot_hold(self) -> None:
        with self.assertRaises(ValueError):
            FixedWeightTernary(_D, _D + 1)


class SqueezeChallengeTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.policy = FixedWeightTernary(_D, _WEIGHT)

    def test_the_same_transcript_state_gives_the_same_challenge(self) -> None:
        _, left = squeeze_challenge(_transcript(), b"c", self.policy)
        _, right = squeeze_challenge(_transcript(), b"c", self.policy)
        np.testing.assert_array_equal(left, right)

    def test_the_label_separates_challenges(self) -> None:
        _, left = squeeze_challenge(_transcript(), b"c", self.policy)
        _, right = squeeze_challenge(_transcript(), b"c2", self.policy)
        self.assertFalse(np.array_equal(left, right))

    def test_prior_observations_separate_challenges(self) -> None:
        _, left = squeeze_challenge(_transcript(b"a"), b"c", self.policy)
        _, right = squeeze_challenge(_transcript(b"b"), b"c", self.policy)
        self.assertFalse(np.array_equal(left, right))

    def test_the_caller_transcript_is_left_untouched(self) -> None:
        original = _transcript()
        squeeze_challenge(original, b"c", self.policy)
        _, again = squeeze_challenge(original, b"c", self.policy)
        _, fresh = squeeze_challenge(_transcript(), b"c", self.policy)
        np.testing.assert_array_equal(again, fresh)

    def test_the_advanced_transcript_carries_the_draw(self) -> None:
        advanced, _ = squeeze_challenge(_transcript(), b"c", self.policy)
        _, after = squeeze_challenge(advanced, b"c", self.policy)
        _, first = squeeze_challenge(_transcript(), b"c", self.policy)
        self.assertFalse(np.array_equal(after, first))


class ChallengePolicyConformanceTest(absltest.TestCase):
    def test_the_sparse_instance_conforms(self) -> None:
        self.assertIsInstance(FixedWeightTernary(_D, _WEIGHT), ChallengePolicy)

    def test_the_lnp_challenge_set_conforms_without_knowing_the_seam(self) -> None:
        policy = ChallengeParams(d=_D, kappa=2, eta=59, k=32)
        self.assertIsInstance(policy, ChallengePolicy)
        # Structural conformance alone would pass on a policy whose parser
        # reads a different count than `bytes_needed` quotes; only a draw
        # through `squeeze_challenge` exercises that pairing.
        _, challenge = squeeze_challenge(_transcript(), b"c", policy)
        self.assertLen(challenge, _D)


if __name__ == "__main__":
    absltest.main()
