# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared Fiat-Shamir challenge-field policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array
from zk_dtypes import efinfo

from zorch.transcript import TranscriptT, reinterpret_challenge


def challenge_limbs(dtype: Any) -> int:
    """Base-field squeezes required to construct one element of ``dtype``."""
    try:
        return efinfo(dtype).degree
    except ValueError:
        return 1


@dataclass(frozen=True)
class ChallengePolicy:
    """The field Fiat-Shamir challenges are drawn in.

    Explicit because it is a soundness parameter: an extension challenge raises
    the soundness floor over a base-field one. Naming it also *promotes* — an
    extension policy draws extension challenges against a base-field claim.

    Naming the transcript's own field costs nothing: ``reinterpret_challenge``
    is then the identity, so it is the one-squeeze schedule.
    """

    dtype: Any

    @property
    def base_limbs(self) -> int:
        """Words per challenge over the challenge field's own base field, for
        sites fixing a static jit-zone width before a transcript is in hand."""
        return challenge_limbs(self.dtype)

    def limbs_over(self, transcript_field: Any) -> int:
        """Transcript words per challenge: the degree ratio, so an
        extension-native sponge spends one word where a base-field sponge
        spends the extension's degree."""
        words = challenge_limbs(self.dtype)
        per_word = challenge_limbs(transcript_field)
        if words % per_word:
            raise ValueError(
                f"{self.dtype} does not tile {transcript_field}: "
                f"{words} coefficients over words of {per_word}"
            )
        return words // per_word

    def _regroup(self, raw: Array, count: int, limbs: int) -> Array:
        """Read ``count`` challenges out of consecutive transcript words."""
        return fnp.stack(
            [
                reinterpret_challenge(raw[i * limbs : (i + 1) * limbs], self.dtype)
                for i in range(count)
            ]
        )

    def sample(self, transcript: TranscriptT) -> tuple[TranscriptT, Array]:
        transcript, challenges = self.sample_many(transcript, 1)
        return transcript, challenges[0]

    def sample_many(
        self, transcript: TranscriptT, count: int
    ) -> tuple[TranscriptT, Array]:
        # One squeeze call for all `count * limbs` limbs: the duplex squeezes a
        # rate-block per permutation, so batching costs ~ceil(n/rate) permutes
        # where a per-challenge loop costs ~n, for the same stream.
        limbs = self.limbs_over(transcript.field)
        transcript, raw = transcript.sample(count * limbs)
        return transcript, self._regroup(raw, count, limbs)

    def observe_and_sample_many(
        self, transcript: TranscriptT, values: Array, count: int
    ) -> tuple[TranscriptT, Array]:
        """Absorb `values` and squeeze `count` challenges in ONE transcript hop.

        The many-challenge form of `observe_and_sample`, for the same reason
        `sample_many` exists and for one more: a hop is the unit that costs. An
        eager hop -- one outside any traced region, as at a stage boundary -- runs
        ~75us against 0.43us of hashing, nearly all of it dispatch, so a head that
        spells this as observe, observe, sample pays three of them for one
        Fiat-Shamir step. Absorbing a concatenation is byte-identical to absorbing
        the parts in order (the sponge appends to its rate buffer), so a caller
        with several messages should join them rather than absorb each."""
        limbs = self.limbs_over(transcript.field)
        transcript, raw = transcript.observe_and_sample(values, count * limbs)
        return transcript, self._regroup(raw, count, limbs)

    def observe_and_sample(
        self, transcript: TranscriptT, values: Array
    ) -> tuple[TranscriptT, Array]:
        """Absorb `values` and squeeze one challenge of this policy's dtype — the
        FS entry point whenever the caller wants a *typed field* challenge. The
        limbs<->dtype packing lives here and nowhere else, so a prover and its
        verifier cannot disagree about it; a caller that wants raw squeezes in the
        transcript's own field calls `transcript.observe_and_sample` directly."""
        transcript, challenges = self.observe_and_sample_many(transcript, values, 1)
        return transcript, challenges[0]
