# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Homogeneous IOP rounds and their recurrence drivers.

A round is one directional step of a repeated protocol recurrence. The generic
contract treats data entering and leaving the step uniformly: prover rounds
receive ``None`` and emit proof messages; verifier rounds receive those messages
and emit validity flags. Stages pair the two executions at protocol boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

from frx import Array

from zorch.transcript import Transcript

Carry = TypeVar("Carry")
Incoming = TypeVar("Incoming", contravariant=True)
Outgoing = TypeVar("Outgoing", covariant=True)


class Round(Protocol[Carry, Incoming, Outgoing]):
    """One recurrence transition with explicit incoming and outgoing data."""

    def __call__(
        self,
        carry: Carry,
        transcript: Transcript,
        incoming: Incoming,
        /,
    ) -> tuple[Carry, Transcript, Outgoing]: ...


# Readable specializations of the single Round protocol. These are aliases, not
# separate interfaces.
ProverRound: TypeAlias = Round[Any, None, Any]
VerifierRound: TypeAlias = Round[Any, Any, Array]
InnerVerifierRound: TypeAlias = Round[Array, Array, tuple[Array, Array]]


def prove_rounds(
    rounds: Iterable[ProverRound], carry: Any, transcript: Transcript
) -> tuple[Any, Transcript, list[Any]]:
    """Run prover rounds, passing no incoming message and collecting outputs."""
    msgs = []
    for rnd in rounds:
        carry, transcript, msg = rnd(carry, transcript, None)
        msgs.append(msg)
    return carry, transcript, msgs


def verify_rounds(
    rounds: Iterable[VerifierRound],
    carry: Any,
    msgs: Sequence[Any],
    transcript: Transcript,
) -> tuple[Any, Transcript, Array]:
    """Replay verifier rounds and aggregate their outgoing verdicts."""
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
