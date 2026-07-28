# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito Fiat-Shamir choreography — the ligerito-specific deltas layered on
the shared `FoldChoreography` base (`zorch/pcs/fold.py`): the terminal
residual binding (`observe_residual`) and the wire's structural message count
(`num_messages`). The shared FS surface (statement binding, round-message
framing, grinding, query sampling) lives on the base; only ligerito's deltas
live here — zorch's native wire as the default behavior, the `alpha_lsb_first`
/ `compressed_sumcheck_messages` config-knob philosophy (one definition, both
sides derive) lifted from data to behavior, for a byte-fixed consumer like
flock's `pcs::ligerito`.

`LigeritoProver` and `LigeritoVerifier` must share ONE choreography instance
(the base's contract: every hook is side-neutral except the grind/check pair,
whose schedule both sides read off the same bits methods).

The message emission policy (`FoldChoreography.eager_messages`) is the
structural choice `num_messages` accounts for. Lazy (default): each round
message is absorbed fused with its challenge squeeze, and the wire carries
exactly one message per fold round. Eager (flock's shape): the current
state's round message is absorbed the moment the state forms — once before
any fold, once after every fold (including the terminal residual state), and
once per introduced basis (each OOD block and each level's induce), always
BEFORE the glue challenge — so the wire also carries those introduce
messages, and the verifier recombines them linearly (round messages are
linear in the basis factor) into the running round poly instead of reading it
whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from frx import Array

from zorch.pcs.fold import FoldChoreography
from zorch.transcript import TranscriptT

if TYPE_CHECKING:
    from zorch.pcs.ligerito.config import LigeritoConfig


@dataclass(frozen=True)
class LigeritoChoreography(FoldChoreography[TranscriptT], Generic[TranscriptT]):
    """zorch's native Ligerito wire as an overridable choreography. Stateless —
    a consumer subclasses and overrides only its deltas."""

    def observe_residual(self, transcript: TranscriptT, residual: Array) -> TranscriptT:
        """Absorb the final level's in-clear residual (a byte wire may frame it
        element by element)."""
        return transcript.observe(residual)

    def num_messages(self, config: LigeritoConfig) -> int:
        """Sumcheck messages on the wire for `config` under this choreography —
        the verifier's structural pre-check. Lazy: one per fold round. Eager:
        plus the initial state's, plus one per introduced basis (each non-final
        level's induce and every OOD block)."""
        n = sum(config.fold_ks)
        if self.eager_messages:
            n += 1 + (config.num_levels - 1) + config.total_ood
        return n

    def num_pow_witnesses(self, config: LigeritoConfig) -> int:
        """Proof-of-work witnesses on the wire — one per scheduled grind, in
        schedule order. Derived from the bits methods so the wire shape cannot
        drift from the schedule."""
        folds = sum(
            1
            for level in range(config.num_levels)
            for j in range(config.fold_ks[level])
            if self.fold_grind_bits(level, j) is not None
        )
        queries = sum(
            1
            for level in range(config.num_levels)
            if self.query_grind_bits(level) is not None
        )
        return folds + queries
