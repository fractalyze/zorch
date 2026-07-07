# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito Fiat-Shamir choreography — the seam that fixes WHEN the recursive
open touches the transcript, decoupled from WHAT the recursion computes.

Two provers can run the identical recursion (same folds, same commits, same
induced bases) and still produce different byte streams: one binds the opening
point, the other binds only the claim; one observes a level's roots and
residual with its own framing; one derives query indices by rejection sampling
instead of a plain reduction. `FsChoreography` owns exactly those choices as
overridable hooks operating on the generic `Transcript`, with zorch's native
wire as the default behavior — the `alpha_lsb_first` /
`compressed_sumcheck_messages` config-knob philosophy (one definition, both
sides derive) lifted from data to behavior, for a byte-fixed consumer like
flock's `pcs::ligerito` (fractalyze/flock-zorch#32).

`LigeritoProver` and `LigeritoVerifier` must share ONE choreography instance:
every hook is side-neutral (a pure transcript interaction), so a shared
instance makes the two Fiat-Shamir streams equal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jax import Array

from zorch.pcs.fold import sample_positions
from zorch.transcript import TranscriptT

if TYPE_CHECKING:
    from zorch.pcs.ligerito.config import LigeritoConfig


@dataclass(frozen=True)
class FsChoreography:
    """zorch's native Ligerito wire as an overridable choreography. Stateless —
    a consumer subclasses and overrides only its deltas."""

    def bind_statement(
        self, transcript: TranscriptT, root: Array, point: Array, value: Array
    ) -> TranscriptT:
        """Bind the opening statement before any challenge. Default binds all
        of (root, point, value) in that order; a consumer whose outer protocol
        already binds the point (flock: through the basis) overrides."""
        transcript = transcript.observe(root)
        transcript = transcript.observe(point)
        return transcript.observe(value)

    def fold_challenge(
        self, transcript: TranscriptT, msg: Array, level: int, fold_idx: int
    ) -> tuple[TranscriptT, Array]:
        """The per-round Fiat-Shamir hop: absorb the round message, squeeze the
        scalar fold challenge. Default is the fused `observe_and_sample` (one
        kernel under `@jit` — the repo's fusion contract)."""
        del level, fold_idx  # the default schedule is position-independent
        transcript, r = transcript.observe_and_sample(msg, 1)
        return transcript, r[0]

    def observe_root(self, transcript: TranscriptT, root: Array) -> TranscriptT:
        """Absorb a recursive level's commit root."""
        return transcript.observe(root)

    def observe_residual(self, transcript: TranscriptT, residual: Array) -> TranscriptT:
        """Absorb the final level's in-clear residual (a byte wire may frame it
        element by element)."""
        return transcript.observe(residual)

    def sample_queries(
        self, transcript: TranscriptT, block_len: int, count: int
    ) -> tuple[TranscriptT, Array]:
        """Squeeze a level's `count` query positions in `[0, block_len)`."""
        return sample_positions(transcript, block_len, count)

    def num_messages(self, config: LigeritoConfig) -> int:
        """Sumcheck messages on the wire for `config` under this choreography —
        the verifier's structural pre-check: one per fold round."""
        return sum(config.fold_ks)
