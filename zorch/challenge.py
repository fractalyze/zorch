# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared Fiat-Shamir challenge-field policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array
from zk_dtypes import efinfo

from zorch.transcript import Transcript, reinterpret_challenge


def challenge_limbs(dtype: Any) -> int:
    """Base-field squeezes required to construct one element of ``dtype``."""
    try:
        return efinfo(dtype).degree
    except ValueError:
        return 1


@dataclass(frozen=True)
class ChallengePolicy:
    """The field Fiat-Shamir challenges are drawn in.

    One knob, always explicit. The field is a soundness parameter — an
    extension challenge raises the soundness floor over a base-field one — so
    it is stated at construction rather than inherited from whatever value a
    round happens to hold. That also lets a claim be *promoted*: a policy over
    an extension field draws extension challenges against a base-field claim,
    which a rule reading the field off the running value cannot express.

    The transcript squeezes base-field words; ``limbs`` is how many of them
    make one challenge. ``reinterpret_challenge`` is the identity when
    ``dtype`` is the transcript's own field, so a base-field policy costs
    exactly the transcript-native one-squeeze schedule.
    """

    dtype: Any

    @property
    def base_limbs(self) -> int:
        """Words per challenge over the challenge field's OWN base field.

        For sites that must fix a static width before a transcript is in hand
        (a jit-zone operand). Equivalent to ``limbs_over`` against a base-field
        sponge, which is what every production transcript is.
        """
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

    def sample(self, transcript: Transcript) -> tuple[Transcript, Array]:
        transcript, challenges = self.sample_many(transcript, 1)
        return transcript, challenges[0]

    def sample_many(
        self, transcript: Transcript, count: int
    ) -> tuple[Transcript, Array]:
        # One squeeze call for all `count * limbs` limbs: the duplex squeezes a
        # rate-block per permutation, so batching costs ~ceil(n/rate) permutes
        # where a per-challenge loop costs ~n, for the same stream.
        limbs = self.limbs_over(transcript.field)
        transcript, raw = transcript.sample(count * limbs)
        return transcript, self._regroup(raw, count, limbs)

    def observe_and_sample(
        self, transcript: Transcript, values: Array
    ) -> tuple[Transcript, Array]:
        # The absorb and the squeeze stay one hop: splitting them into `observe`
        # then `sample` bypasses the duplex-FS fusion marker and scatters ~9
        # kernels per round, which `zorch`'s fusion rule does not allow.
        limbs = self.limbs_over(transcript.field)
        transcript, raw = transcript.observe_and_sample(values, limbs)
        return transcript, self._regroup(raw, 1, limbs)[0]
