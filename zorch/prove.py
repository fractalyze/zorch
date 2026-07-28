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

from zorch.transcript import TranscriptT

if TYPE_CHECKING:
    from zorch.round import ProverRound


def fold_rounds(
    rnd: ProverRound[Any, Any, TranscriptT],
    state: Any,
    transcript: TranscriptT,
    rounds: int,
) -> tuple[Any, TranscriptT, list[Any]]:
    """Run `rnd` exactly `rounds` times; return (state, transcript, list[msg])."""
    msgs: list[Any] = []
    for _ in range(rounds):
        state, transcript, msg = rnd(state, transcript)
        msgs.append(msg)
    return state, transcript, msgs
