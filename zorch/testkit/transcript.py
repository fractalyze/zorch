# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""A fast, deterministic transcript for unit tests.

`cheap_transcript` returns a real `DuplexTranscript` over `CheapPermutation` — a
genuine Fiat-Shamir sponge whose challenges derive from observations, but cheap
enough for unit tests and gated `fusion_path=GENERIC` so `prove` keeps it
on the unmarked path (no `zorch.sumcheck` marker).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
from frx import Array
from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath

from zorch.transcript import DuplexTranscript

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation


class CheapPermutation:
    """A deterministic, NON-cryptographic fixed-width permutation for tests only.

    Implements the `Permutation` seam so a `DuplexTranscript` runs a real
    (sound-shaped) Fiat-Shamir sponge in unit tests without a full poseidon2
    permute. `fusion_path` is GENERIC, so `prove` keeps such a transcript
    on its unmarked path (no `zorch.sumcheck` marker). The mixing is not a secure
    permutation and need not be bijective — never use outside tests.
    """

    def __init__(self, width: int, dtype: Any) -> None:
        self.width = width
        self.dtype = dtype
        # Keep test transcripts on prove's unmarked path.
        self.fusion_path = FusionPath.GENERIC
        self.fused_region_marker = (FUSED_REGION_MARKER, 0)

    def __eq__(self, other: object) -> bool:
        # Pytree-aux value equality, mirroring `Poseidon2`
        # (docs/reference/conventions.md "Pytree registration") — identity-eq
        # would re-trace per fresh instance.
        if not isinstance(other, CheapPermutation):
            return NotImplemented
        return (self.width, self.dtype) == (other.width, other.dtype)

    def __hash__(self) -> int:
        return hash((self.width, self.dtype))

    def permute(self, state: Array) -> Array:
        # Cube each lane, then fold the lane-sum back into every lane so each
        # output diffuses all of the absorbed state (the squeezed lane must see
        # every observed lane). Enough for distinct, deterministic per-round
        # challenges; not a secure or even bijective permutation — tests only.
        mixed = state * state * state
        return mixed + fnp.sum(mixed)

    # Inert fused-region ABI: non-fused, never called; a conformant stub.
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        return (leading,), (lambda state, *ops: self.permute(state)), {}


def cheap_transcript(dtype: Any, *, width: int = 8, rate: int = 4) -> DuplexTranscript:
    """A fresh `DuplexTranscript` over `CheapPermutation` for tests — a real sponge
    whose challenges derive from observations (no preset stream)."""
    return DuplexTranscript.new(CheapPermutation(width=width, dtype=dtype), rate=rate)


if TYPE_CHECKING:
    _: type[Permutation] = CheapPermutation
