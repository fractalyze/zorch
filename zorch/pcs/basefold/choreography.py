# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold Fiat-Shamir choreography — the seam that fixes WHEN the interleaved
sumcheck + FRI fold touches the transcript, decoupled from WHAT it computes.

`BasefoldChoreography` owns the shared Fiat-Shamir surface (statement binding,
round-message framing, grinding, query sampling) as overridable hooks, plus two
basefold-specific FRAMING hooks: how the round-message components are stacked
onto the wire (`round_message`) and the terminal binding (`observe_final`). The
round ALGEBRA — the message components themselves, the state fold, and the
verifier's claim recurrence — lives on the `SumcheckKernel` seam (`kernel.py`),
not here. zorch's native BaseFold wire is the default behavior, derived from
`BasefoldProver.open`/`BasefoldVerifier.verify` — a byte-fixed consumer
subclasses and overrides only its deltas, the `LigeritoChoreography` pattern
applied to BaseFold. Standalone for now (no shared base yet): the shared hooks
mirror `LigeritoChoreography`'s verbatim, and the two collapse onto one
`FoldChoreography` base once this class is proven against a second wire.

`BasefoldProver` and `BasefoldVerifier` must share ONE choreography instance:
every hook is side-neutral (a pure transcript interaction) except the
grind/check pair, whose schedule both sides read off the same bits methods, so
a shared instance keeps the two Fiat-Shamir streams equal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
from jax import Array

from zorch.pcs.fold import sample_positions
from zorch.transcript import GrindingTranscript, TranscriptT

if TYPE_CHECKING:
    from zorch.pcs.basefold.config import BasefoldConfig


@dataclass(frozen=True)
class BasefoldChoreography:
    """zorch's native BaseFold wire as an overridable choreography. Stateless —
    a consumer subclasses and overrides only its deltas."""

    # --- shared FS surface (mirrors `LigeritoChoreography` verbatim; folds
    # onto a shared `FoldChoreography` base once one exists) ---

    @property
    def eager_messages(self) -> bool:
        """False: round messages ride fused observe+sample hops
        (`fold_challenge`). True: `observe_message` absorbs each message at
        emission time and `fold_challenge` must be overridden to a bare sample
        (its `msg` arrives as None) — the two are one policy, split only so the
        driver can place the interactions."""
        return False

    def bind_statement(
        self, transcript: TranscriptT, root: Array, point: Array | None, value: Array
    ) -> TranscriptT:
        """Bind the opening statement before any challenge. Default binds all
        of (root, point, value) in that order; a consumer whose outer protocol
        already binds the point overrides. `point` is None under a raw-basis
        entry, where no point exists — the native binding refuses rather than
        silently bind less."""
        transcript = transcript.observe(root)
        if point is None:
            raise ValueError(
                "the native statement binding observes the opening point, but "
                "this entry carries none — a basis-entry consumer must "
                "override bind_statement (the basis binds the statement)"
            )
        transcript = transcript.observe(point)
        return transcript.observe(value)

    def observe_message(self, transcript: TranscriptT, msg: Array) -> TranscriptT:
        """Absorb one eagerly emitted message (eager policy only)."""
        return transcript.observe(msg)

    def fold_challenge(
        self, transcript: TranscriptT, msg: Array | None, level: int, fold_idx: int
    ) -> tuple[TranscriptT, Array]:
        """The per-round Fiat-Shamir hop: absorb the round message, squeeze the
        scalar fold challenge. Default is the fused `observe_and_sample` (one
        kernel under `@jit` — the repo's fusion contract). Under the eager
        policy `msg` is None (already absorbed) and the override samples bare."""
        del level, fold_idx  # the default schedule is position-independent
        if msg is None:
            raise ValueError(
                "the lazy default absorbs the round message here; an eager "
                "choreography must override fold_challenge to a bare sample"
            )
        transcript, r = transcript.observe_and_sample(msg, 1)
        return transcript, r[0]

    def observe_root(self, transcript: TranscriptT, root: Array) -> TranscriptT:
        """Absorb a fold round's pre-fold pair-commit root."""
        return transcript.observe(root)

    def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
        """Proof-of-work schedule for a fold round, ground between the round
        message's absorb and its challenge squeeze. None (default) = no grind
        and nothing on the wire; an int puts a witness on the wire — 0 included
        (a 0-bit grind is trivial but still advances the transcript)."""
        del level, fold_idx
        return None

    def query_grind_bits(self, level: int) -> int | None:
        """Proof-of-work schedule for the query phase, ground right before its
        positions are sampled. Same None / int-including-0 contract as
        `fold_grind_bits`."""
        del level
        return None

    def grind(self, transcript: TranscriptT, bits: int) -> tuple[TranscriptT, Array]:
        """Prover-side grind (called only when the bits schedule says so).
        Default is the `GrindingTranscript` seam, so a zorch-native consumer
        adds grinding by overriding only the bits methods; a byte-wire consumer
        overrides the mechanism too."""
        grinding = cast(GrindingTranscript, transcript)
        advanced, witness = grinding.grind(bits)
        return cast(TranscriptT, advanced), witness

    def check_grind(
        self, transcript: TranscriptT, bits: int, witness: Array
    ) -> tuple[TranscriptT, Array]:
        """Verifier-side dual of `grind`: replay the witness, return
        `(transcript, ok)` with the transcript advanced identically."""
        grinding = cast(GrindingTranscript, transcript)
        advanced, ok = grinding.check_witness(bits, witness)
        return cast(TranscriptT, advanced), ok

    def sample_queries(
        self, transcript: TranscriptT, block_len: int, count: int
    ) -> tuple[TranscriptT, Array]:
        """Squeeze `count` query positions in `[0, block_len)`."""
        return sample_positions(transcript, block_len, count)

    def num_pow_witnesses(self, config: BasefoldConfig) -> int:
        """Proof-of-work witnesses on the wire — one per scheduled grind, in
        schedule order. BaseFold's flat schedule has one grind slot per fold
        round (`config.num_vars` of them) and one for the single query phase;
        derived from the bits methods so the wire shape cannot drift from the
        schedule."""
        folds = sum(
            1
            for round_idx in range(config.num_vars)
            if self.fold_grind_bits(round_idx, 0) is not None
        )
        queries = 1 if self.query_grind_bits(0) is not None else 0
        return folds + queries

    # --- basefold-specific ---

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
