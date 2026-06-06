# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic multi-round proving driver.

`fold_rounds` runs a folding `Round` a fixed number of times, threading the
(state, transcript) carry and collecting each round's message into a list — the
state and message types are opaque to the driver, so any scheme reuses it: a
Basefold round (univariate + commitment message), a multilinear sumcheck round, a
future univariate / FFT round. It stays a Python loop: its per-round message
shapes need not be round-invariant, so it is not `lax.scan`-shaped.

The homogeneous multilinear sumcheck specialization — one `lax.scan` over the
hypercube variables, with the optional `zorch.sumcheck` register-resident marker —
lives in `zorch.sumcheck.prover`, not here: it is multilinear-specific, where this
driver is basis-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.round import ProverRound


def fold_rounds(
    round: ProverRound, state: Any, transcript: Transcript, rounds: int
) -> tuple[Any, Transcript, list[Any]]:
    """Run `round` exactly `rounds` times; return (state, transcript, list[msg])."""
    msgs: list[Any] = []
    for _ in range(rounds):
        state, transcript, msg = round(state, transcript)
        msgs.append(msg)
    return state, transcript, msgs
