# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generic 2-to-1 folding driver.

Runs a folding `Round` to completion: each call halves the carry (state) and
emits one message, until the state collapses to width 1. Protocol-agnostic —
any IOP whose round folds the state 2-to-1 (sumcheck, FRI, Basefold, a GKR
layer) uses this same driver. The top-level prover is just calling it on the
top `Round` with a fresh transcript.
"""
from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from zorch.round import Round
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


def prove(
    round: Round, state: Sequence[Array], transcript: Transcript
) -> tuple[Sequence[Array], Transcript, Array]:
    """Run `round` once per variable; collect the per-round messages.

    Returns (state, transcript, proof) where proof stacks the messages.
    """
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    n = log2_strict_usize(state[0].shape[-1])
    msgs: list[Array] = []
    for _ in range(n):
        state, transcript, msg = round(state, transcript)
        msgs.append(msg)
    return state, transcript, jnp.stack(msgs)
