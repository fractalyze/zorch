# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold Fiat-Shamir choreography — the basefold-specific deltas layered on
the shared `FoldChoreography` base (`zorch/pcs/fold.py`): how the round-message
components are stacked onto the wire (`round_message`) and the terminal
binding (`observe_final`). The round ALGEBRA — the message components
themselves, the state fold, and the verifier's claim recurrence — lives on the
`SumcheckKernel` seam (`kernel.py`), not here. zorch's native BaseFold wire is
the default behavior, derived from `BasefoldProver.open`/`BasefoldVerifier.verify`
— a byte-fixed consumer subclasses and overrides only its deltas, the
`LigeritoChoreography` pattern applied to BaseFold.

`BasefoldProver` and `BasefoldVerifier` must share ONE choreography instance
(the base's contract: every hook is side-neutral except the grind/check pair,
whose schedule both sides read off the same bits methods).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import frx.numpy as jnp
from frx import Array

from zorch.pcs.fold import FoldChoreography
from zorch.transcript import TranscriptT

if TYPE_CHECKING:
    from zorch.pcs.basefold.config import BasefoldConfig


@dataclass(frozen=True)
class BasefoldChoreography(FoldChoreography):
    """zorch's native BaseFold wire as an overridable choreography. Stateless —
    a consumer subclasses and overrides only its deltas."""

    def num_pow_witnesses(self, config: BasefoldConfig) -> int:
        """Proof-of-work witnesses on the wire — one per scheduled grind, in
        schedule order. BaseFold's flat schedule has one grind slot per fold
        round (`config.num_vars` of them) and one for the single query phase;
        derived from the bits methods so the wire shape cannot drift from the
        schedule."""
        folds = sum(
            self.fold_grind_bits(round_idx, 0) is not None
            for round_idx in range(config.num_vars)
        )
        queries = 1 if self.query_grind_bits(0) is not None else 0
        return folds + queries

    def round_message(self, *components: Array) -> Array:
        """Frame the kernel's raw round-message components onto the wire: stack
        them into one array (native: `(s(0), s(1))`; a product consumer:
        `(u0, u2)`). The FRAMING only — the components come from the kernel, and
        a consumer whose transcript absorbs them element-by-element (e.g. two
        scalar observes) keeps this default because `observe_message` iterates
        the stacked array under such a transcript."""
        return jnp.stack(list(components))

    def observe_final(self, transcript: TranscriptT, final_poly: Array) -> TranscriptT:
        """Bind the terminal poly before sampling queries. Default observes
        the whole final codeword in the clear (the IOPP terminal binding)."""
        return transcript.observe(final_poly)
