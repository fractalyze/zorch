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
