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
    """Select the field and squeeze width for Fiat-Shamir challenges.

    The default preserves the transcript-native one-squeeze schedule. ``dtype``
    selects an explicit target field and derives its limb count. ``limbs`` can
    override that count; with no explicit dtype it selects the caller-provided
    value dtype. The latter form supports coefficient and jagged engines whose
    field is carried by the running claim rather than static configuration.
    """

    dtype: Any | None = None
    limbs: int | None = None

    def __post_init__(self) -> None:
        if self.limbs is not None and self.limbs < 1:
            raise ValueError("challenge limbs must be >= 1")

    @property
    def needs_value_dtype(self) -> bool:
        """Whether sampling requires the caller to supply a value dtype.

        A limb-only policy takes its field from the running claim, so a site
        that samples fresh randomness has no value to inherit one from and must
        reject such a policy up front rather than failing mid-proof.
        """
        return self.dtype is None and self.limbs is not None

    def target_dtype(self, value_dtype: Any | None = None) -> Any | None:
        if self.dtype is not None:
            return self.dtype
        if self.limbs is not None:
            if value_dtype is None:
                raise ValueError("a limb-only challenge policy needs a value dtype")
            return value_dtype
        return None

    def limbs_for(self, value_dtype: Any | None = None) -> int:
        target = self.target_dtype(value_dtype)
        if self.limbs is not None:
            return self.limbs
        return 1 if target is None else challenge_limbs(target)

    def _regroup(self, raw: Array, count: int, value_dtype: Any | None) -> Array:
        """Read ``count`` challenges out of consecutive base-field squeezes."""
        target = self.target_dtype(value_dtype)
        if target is None:
            return raw
        limbs = self.limbs_for(value_dtype)
        return fnp.stack(
            [
                reinterpret_challenge(raw[i * limbs : (i + 1) * limbs], target)
                for i in range(count)
            ]
        )

    def sample(
        self, transcript: Transcript, value_dtype: Any | None = None
    ) -> tuple[Transcript, Array]:
        transcript, challenges = self.sample_many(transcript, 1, value_dtype)
        return transcript, challenges[0]

    def sample_many(
        self, transcript: Transcript, count: int, value_dtype: Any | None = None
    ) -> tuple[Transcript, Array]:
        # One squeeze call for all `count * limbs` limbs: the duplex squeezes a
        # rate-block per permutation, so batching costs ~ceil(n/rate) permutes
        # where a per-challenge loop costs ~n, for the same stream.
        transcript, raw = transcript.sample(count * self.limbs_for(value_dtype))
        return transcript, self._regroup(raw, count, value_dtype)

    def observe_and_sample(
        self,
        transcript: Transcript,
        values: Array,
        value_dtype: Any | None = None,
    ) -> tuple[Transcript, Array]:
        # The absorb and the squeeze stay one hop: splitting them into `observe`
        # then `sample` bypasses the duplex-FS fusion marker and scatters ~9
        # kernels per round, which `zorch`'s fusion rule does not allow.
        transcript, raw = transcript.observe_and_sample(
            values, self.limbs_for(value_dtype)
        )
        return transcript, self._regroup(raw, 1, value_dtype)[0]


# The immutable protocol default: use the transcript's ordinary challenge
# sampling without an explicit target field or limb-width override.
DEFAULT_CHALLENGES = ChallengePolicy()
