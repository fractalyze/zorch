# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""2-to-1 folding drivers: a generic Python loop and a scanned sumcheck prover.

`fold_rounds` runs a folding `Round` a fixed number of times, threading the
(state, transcript) carry and collecting each round's message into a list — the
state and message types are opaque to the driver, so a scheme whose message is
structured (Basefold: univariate + commitment) reuses it. It stays a Python loop:
its per-round message shapes need not be round-invariant, so it is not `lax.scan`-
shaped.

`prove` is the homogeneous product-sumcheck driver: the per-variable loop becomes
one `lax.scan` so the whole proof compiles to a single traced region, flat in the
variable count (issue #58) — an unrolled loop would inflate the graph past the ZKX
PTX cliff. The catch is that a `scan` carry must keep a fixed shape, but the MLE
state halves every round. So the carry holds each factor in a full-width buffer
with the live data packed at the front: the round reads the live first/second
halves (`buf[..., :N//2]` and a `lax.dynamic_slice` at the live split point),
masks the dead tail out of the round-poly sum, and folds back into the front,
zero-padding the rest. The dead tail is never read as live data, so it cannot
pollute the result — the proof is byte-identical to the Python-loop fold.
"""
from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import reduce
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.round import Round
from zorch.sumcheck.prover import SumcheckRound, fold_pair, lift_to_domain
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
    round: SumcheckRound, state: Sequence[Array], transcript: Transcript
) -> tuple[list[Array], Transcript, Array]:
    """Scan a product-sumcheck round once per variable; stack the round polys.

    `round` is the product `sumcheck.prover.SumcheckRound`; the generic
    heterogeneous case stays on `fold_rounds`.
    """
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    width = state[0].shape[-1]
    rounds = log2_strict_usize(width)
    if rounds == 0:
        raise ValueError(
            "prove requires a state width >= 2 (at least one round), got "
            f"width {width}"
        )
    degree = round.degree
    half_max = width // 2

    def step(
        carry: tuple[list[Array], Transcript, Array], _: None
    ) -> tuple[tuple[list[Array], Transcript, Array], Array]:
        state, transcript, half = carry
        live = jnp.arange(half_max) < half
        pairs = [
            (buf[..., :half_max], lax.dynamic_slice_in_dim(buf, half, half_max, -1))
            for buf in state
        ]
        prod = reduce(
            operator.mul, (lift_to_domain(p0, p1, degree) for p0, p1 in pairs)
        )
        msg = jnp.sum(jnp.where(live, prod, jnp.zeros((), prod.dtype)), axis=-1)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = [
            jnp.concatenate([fold_pair(p0, p1, r[0]), jnp.zeros_like(p0)], axis=-1)
            for p0, p1 in pairs
        ]
        return (state, transcript, half // 2), msg

    init = (list(state), transcript, jnp.int32(half_max))
    (state, transcript, _), msgs = lax.scan(step, init, xs=None, length=rounds)
    return [buf[..., :1] for buf in state], transcript, msgs
