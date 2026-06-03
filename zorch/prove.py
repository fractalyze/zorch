# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generic 2-to-1 folding driver.

`fold_rounds` runs a folding `Round` a fixed number of times, threading the
(state, transcript) carry and collecting each round's message into a list — the
state and message types are opaque to the driver, so a scheme whose message is
structured (Basefold: univariate + commitment) reuses it. `prove` is the
sumcheck-flavoured wrapper: it derives the round count from the carry width and
stacks the homogeneous messages.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import Array

from zorch.round import Round
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


def fold_rounds(
    round: Round, state: Any, transcript: Transcript, rounds: int
) -> tuple[Any, Transcript, list[Any]]:
    """Run `round` exactly `rounds` times; return (state, transcript, list[msg])."""
    msgs: list[Any] = []
    for _ in range(rounds):
        state, transcript, msg = round(state, transcript)
        msgs.append(msg)
    return state, transcript, msgs


def prove(
    round: Round, state: Sequence[Array], transcript: Transcript
) -> tuple[Sequence[Array], Transcript, Array]:
    """Run a folding round once per variable; stack the per-round messages."""
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    rounds = log2_strict_usize(state[0].shape[-1])
    if rounds == 0:
        raise ValueError(
            "prove requires a state width >= 2 (at least one round), got "
            f"width {state[0].shape[-1]}"
        )
    state, transcript, msgs = fold_rounds(round, state, transcript, rounds)
    return state, transcript, jnp.stack(msgs)
