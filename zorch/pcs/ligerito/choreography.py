# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito Fiat-Shamir choreography — the seam that fixes WHEN the recursive
open touches the transcript, decoupled from WHAT the recursion computes.

Two provers can run the identical recursion (same folds, same commits, same
induced bases) and still produce different byte streams: one binds the opening
point, the other binds only the claim; one grinds a proof-of-work between a
round message and its challenge; one observes each state's round message the
moment the state forms, the other fuses observe+sample at round start; one
derives query indices by rejection sampling instead of a plain reduction.
`LigeritoChoreography` owns exactly those choices as overridable hooks operating on
the generic `Transcript`, with zorch's native wire as the default behavior —
the `alpha_lsb_first` / `compressed_sumcheck_messages` config-knob philosophy
(one definition, both sides derive) lifted from data to behavior, for a
byte-fixed consumer like flock's `pcs::ligerito` (fractalyze/flock-zorch#32).

`LigeritoProver` and `LigeritoVerifier` must share ONE choreography instance:
every hook is side-neutral (a pure transcript interaction) except the
grind/check pair, whose schedule both sides read off the same bits methods, so
a shared instance keeps the two Fiat-Shamir streams equal by construction.

The message emission policy is the structural choice. Lazy (default): each
round message is absorbed fused with its challenge squeeze, and the wire
carries exactly one message per fold round. Eager (flock's shape): the current
state's message is absorbed the moment the state forms — once before any fold,
once after every fold (including the terminal residual state), and once per
introduced basis (each OOD block and each level's induce), always BEFORE the
glue challenge — so the wire also carries those introduce messages, and the
verifier recombines them linearly (round messages are linear in the basis
factor) into the running round poly instead of reading it whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from jax import Array

from zorch.pcs.fold import sample_positions
from zorch.transcript import GrindingTranscript, TranscriptT

if TYPE_CHECKING:
    from zorch.pcs.ligerito.config import LigeritoConfig


@dataclass(frozen=True)
class LigeritoChoreography:
    """zorch's native Ligerito wire as an overridable choreography. Stateless —
    a consumer subclasses and overrides only its deltas."""

    @property
    def eager_messages(self) -> bool:
        """False: round messages ride fused observe+sample hops
        (`fold_challenge`). True: `observe_message` absorbs each message at
        emission time and `fold_challenge` must be overridden to a bare sample
        (its `msg` arrives as None) — the two are one policy, split only so the
        driver can place the interactions."""
        return False

    def bind_statement(
        self, transcript: TranscriptT, root: Array, point: Array, value: Array
    ) -> TranscriptT:
        """Bind the opening statement before any challenge. Default binds all
        of (root, point, value) in that order; a consumer whose outer protocol
        already binds the point (flock: through the basis) overrides."""
        transcript = transcript.observe(root)
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
        """Absorb a recursive level's commit root."""
        return transcript.observe(root)

    def observe_residual(self, transcript: TranscriptT, residual: Array) -> TranscriptT:
        """Absorb the final level's in-clear residual (a byte wire may frame it
        element by element)."""
        return transcript.observe(residual)

    def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
        """Proof-of-work schedule for a fold round, ground between the round
        message's absorb and its challenge squeeze. None (default) = no grind
        and nothing on the wire; an int puts a witness on the wire — 0 included
        (a 0-bit grind is trivial but still advances the transcript, flock's
        unconditional-nonce convention)."""
        del level, fold_idx
        return None

    def query_grind_bits(self, level: int) -> int | None:
        """Proof-of-work schedule for a level's query phase, ground right
        before its positions are sampled. Same None / int-including-0 contract
        as `fold_grind_bits`."""
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
        """Squeeze a level's `count` query positions in `[0, block_len)`."""
        return sample_positions(transcript, block_len, count)

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
