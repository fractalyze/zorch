# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared Fiat-Shamir challenge-field policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array
from zk_dtypes import efinfo

from zorch.transcript import Transcript, sample_challenge


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
    def is_native(self) -> bool:
        return self.dtype is None and self.limbs is None

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

    def sample(
        self, transcript: Transcript, value_dtype: Any | None = None
    ) -> tuple[Transcript, Array]:
        target = self.target_dtype(value_dtype)
        if target is None:
            transcript, sampled = transcript.sample(1)
            return transcript, sampled[0]
        return sample_challenge(transcript, target, self.limbs_for(value_dtype))

    def sample_many(
        self, transcript: Transcript, count: int, value_dtype: Any | None = None
    ) -> tuple[Transcript, Array]:
        if self.is_native:
            return transcript.sample(count)
        values = []
        for _ in range(count):
            transcript, value = self.sample(transcript, value_dtype)
            values.append(value)
        return transcript, fnp.stack(values)

    def observe_and_sample(
        self,
        transcript: Transcript,
        values: Array,
        value_dtype: Any | None = None,
    ) -> tuple[Transcript, Array]:
        if self.is_native:
            transcript, sampled = transcript.observe_and_sample(values, 1)
            return transcript, sampled[0]
        return self.sample(transcript.observe(values), value_dtype)
