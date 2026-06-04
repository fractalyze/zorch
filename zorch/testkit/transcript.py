# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""A fast, deterministic transcript for tests — the replacement for the removed
`StubTranscript`.

`StubTranscript` fed preset challenges and ignored observations: convenient, but
not a real Fiat-Shamir transcript, and (lacking a `permutation`) it couldn't ride
`prove`'s `has_dedicated_fusion` marker gate. `cheap_transcript` instead returns a
real `DuplexTranscript` over `CheapPermutation` — a genuine sponge whose
challenges derive from observations, just cheap enough for unit tests and gated
`has_dedicated_fusion=False` so `prove` keeps it on the unmarked path.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from zorch.transcript import DuplexTranscript


class CheapPermutation:
    """A deterministic, NON-cryptographic fixed-width permutation for tests only.

    Implements the `Permutation` seam so a `DuplexTranscript` runs a real
    (sound-shaped) Fiat-Shamir sponge in unit tests without a full poseidon2
    permute. `has_dedicated_fusion` is False, so `prove` keeps such a transcript
    on its unmarked path (no `zorch.sumcheck` marker). The mixing is not a secure
    permutation and need not be bijective — never use outside tests.
    """

    def __init__(self, width: int, dtype: Any) -> None:
        self.width = width
        self.dtype = dtype
        # Keep test transcripts on prove's unmarked path.
        self.has_dedicated_fusion = False

    def permute(self, state: Array) -> Array:
        # Cube each lane, then fold the lane-sum back into every lane so each
        # output diffuses all of the absorbed state (the squeezed lane must see
        # every observed lane). Enough for distinct, deterministic per-round
        # challenges; not a secure or even bijective permutation — tests only.
        mixed = state * state * state
        return mixed + jnp.sum(mixed)


def cheap_transcript(dtype: Any, *, width: int = 8, rate: int = 4) -> DuplexTranscript:
    """A fresh `DuplexTranscript` over `CheapPermutation` for tests — drop-in for
    the old `StubTranscript(...)`, minus preset challenges (challenges now derive
    from observations, like a real sponge)."""
    return DuplexTranscript.new(CheapPermutation(width=width, dtype=dtype), rate=rate)
