# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Homogeneous IOP rounds and their recurrence drivers.

A round is one step of a repeated protocol recurrence. The two roles have
genuinely different shapes, so they are separate protocols: a prover round
advances its carry and emits a proof message; a verifier round consumes that
message, advances its own carry, and reports both the protocol data the driver
accumulates and its consistency verdict. Stages pair the two at protocol
boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from frx import Array
from frx.tree_util import register_dataclass

from zorch.transcript import TranscriptT

Carry = TypeVar("Carry")
Message_co = TypeVar("Message_co", covariant=True)
Message_contra = TypeVar("Message_contra", contravariant=True)


@register_dataclass
@dataclass(frozen=True)
class RunningClaim:
    """A partially built evaluation claim: the value after the rounds bound so far.

    `index` is the write cursor into `point`; both are fixed-shape so the whole
    carry stays one shape across the loop.
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


class ProverRound(Protocol[Carry, Message_co, TranscriptT]):
    """One prover recurrence step: fold the carry and emit a proof message."""

    def __call__(
        self,
        carry: Carry,
        transcript: TranscriptT,
        /,
    ) -> tuple[Carry, TranscriptT, Message_co]: ...


class VerifierRound(Protocol[Carry, Message_contra, TranscriptT]):
    """The dual step: consume the message, fold the carry, report a verdict.

    Anything the round derives rather than receives — a sumcheck round's
    challenge, a fold challenge — belongs in the carry, not in a second return
    slot. Only the message crosses between roles, so only the message is a
    separate position, and every recurrence shape shares this one contract.
    """

    def __call__(
        self,
        carry: Carry,
        transcript: TranscriptT,
        message: Message_contra,
        /,
    ) -> tuple[Carry, TranscriptT, Array]: ...


def prove_rounds(
    rounds: Iterable[ProverRound[Any, Any, TranscriptT]],
    carry: Any,
    transcript: TranscriptT,
) -> tuple[Any, TranscriptT, list[Any]]:
    """Run prover rounds, collecting each step's message."""
    msgs = []
    for rnd in rounds:
        carry, transcript, msg = rnd(carry, transcript)
        msgs.append(msg)
    return carry, transcript, msgs


def verify_rounds(
    rounds: Iterable[VerifierRound[Any, Any, TranscriptT]],
    carry: Any,
    msgs: Sequence[Any],
    transcript: TranscriptT,
) -> tuple[Any, TranscriptT, Array]:
    """Replay verifier rounds over a heterogeneous message list, ANDing verdicts.

    The sibling of `zorch.verify.verify`, which scans one round over a dense
    proof array; both consume the same protocol.
    """
    materialized = list(rounds)
    if len(msgs) != len(materialized):
        raise ValueError(
            f"need one message per round: {len(materialized)} rounds, "
            f"got {len(msgs)} messages"
        )
    ok: Any = True
    for rnd, msg in zip(materialized, msgs, strict=True):
        carry, transcript, ok_round = rnd(carry, transcript, msg)
        ok = ok & ok_round
    return carry, transcript, ok
