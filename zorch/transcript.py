# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fiat-Shamir transcript interface + a test/dev stub.

The real duplex-sponge transcript (issue #3) implements the same Protocol and
drops in. `StubTranscript` yields preset challenges and ignores observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

from jax import Array


class Transcript(Protocol):
    def observe(self, values: Array) -> Self: ...
    def sample(self, n: int = 1) -> tuple[Self, Array]: ...


@dataclass(frozen=True)
class StubTranscript:
    """Preset challenge stream; `observe` is a no-op. Test/dev only."""

    challenges: Array  # (k,) preset challenge values
    pos: int = 0  # index of the next challenge

    def observe(self, values: Array) -> "StubTranscript":
        return self

    def sample(self, n: int = 1) -> tuple["StubTranscript", Array]:
        out = self.challenges[self.pos : self.pos + n]
        return StubTranscript(self.challenges, self.pos + n), out
