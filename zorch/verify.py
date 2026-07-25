# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck replay driver — the dual of `prove`.

The per-round check and claim-reduction live on the verifier round; this driver
only scans it over the proof, threads the transcript, and ANDs the consistency
flags. It stops at the reduced point-claim — the final
`final_claim == oracle(point)` check needs a PCS opening and is the consumer's,
keeping this block proving-scheme- and PCS-agnostic.

The bound point rides the carry in a preallocated buffer rather than the scan's
per-step output channel, so a round reports only its verdict and the round
protocol stays the same shape as one that exports nothing (a GKR layer). The
challenge is derived state, not a message, and derived state belongs in the
carry.

The replay is one `lax.scan` over the proof rows, not a Python loop: the whole
verification compiles to a single traced region that is flat in the round count, so it
stays one fused unit rather than an unrolled body that
crosses the XLA PTX cliff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array, lax
from frx.tree_util import register_dataclass

from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.round import VerifierRound


@register_dataclass
@dataclass(frozen=True)
class RunningClaim:
    """A partially built evaluation claim: the value after the rounds bound so far.

    `index` is the write cursor into `point`; both are fixed-shape so the whole
    carry is a valid `lax.scan` carry.
    """

    value: Array
    point: Array
    index: Array

    def bind(self, value: Array, challenge: Array) -> RunningClaim:
        """Advance to the reduced claim, recording this round's challenge.

        The single definition of the write, so every wire form records its
        challenge identically and a new one cannot get the bookkeeping wrong.
        """
        return RunningClaim(
            value, self.point.at[self.index].set(challenge), self.index + 1
        )


def verify(
    verifier: VerifierRound[RunningClaim, Array],
    claim: Array,
    proof: Array,
    transcript: Transcript,
) -> tuple[Array, Array, Transcript, Array]:
    """Replay `proof` against `claim` → `(point, final_claim, transcript, ok)`.

    `ok` ANDs every round's check; one false anywhere rejects the proof.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")

    init = RunningClaim(claim, fnp.zeros((proof.shape[0],), claim.dtype), fnp.int32(0))

    def step(
        carry: tuple[RunningClaim, Transcript], msg: Array
    ) -> tuple[tuple[RunningClaim, Transcript], Array]:
        state, transcript = carry
        state, transcript, ok = verifier(state, transcript, msg)
        return (state, transcript), ok

    (state, transcript), oks = lax.scan(step, (init, transcript), proof)
    return state.point, state.value, transcript, fnp.all(oks)
