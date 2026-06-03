# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""2-to-1 folding drivers: a generic Python loop and a scanned sumcheck prover.

`fold_rounds` runs a folding `Round` a fixed number of times, threading the
(state, transcript) carry and collecting each round's message into a list — the
state and message types are opaque to the driver, so a scheme whose message is
structured (Basefold: univariate + commitment) reuses it. It stays a Python loop:
its per-round message shapes need not be round-invariant, so it is not `lax.scan`-
shaped.

`prove` is the homogeneous sumcheck driver, generic over the round's summand: the
per-variable loop becomes one `lax.scan` so the whole proof compiles to a single
traced region, flat in the variable count (issue #58) — an unrolled loop would
inflate the graph past the ZKX PTX cliff. The driver owns the split / mask / fold /
scan and reads only the round's `degree` + `_combine` (the `SumcheckSummand` seam),
so the product `SumcheckRound` and the LogUp `LogupSumcheckRound` share this one
scan instead of each forking a copy — the summand is the only thing that varies.
The catch is that a `scan` carry must keep a fixed shape, but the MLE state halves
every round. So the carry holds each factor in a full-width buffer with the live
data packed at the front: the step reads the live first/second halves
(`buf[..., :N//2]` and a `lax.dynamic_slice` at the live split point), masks the
dead tail out of the round-poly sum, and folds back into the front, zero-padding
the rest. The dead tail is never read as live data, so it cannot pollute the
result — the proof is byte-identical to the Python-loop fold. Each round's output
is a `RoundMsg` (round poly + sampled challenge); the scan stacks them, so `prove`
hands back one `msgs` whose `.round_poly` is the proof and whose `.challenge` is
the evaluation point — the prover's per-round message stream, the dual of what
`verify` consumes.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import register_dataclass

from zorch.round import ProverRound
from zorch.sumcheck.prover import fold_pair, lift_to_domain
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


@partial(register_dataclass, data_fields=["round_poly", "challenge"], meta_fields=[])
@dataclass(frozen=True)
class RoundMsg:
    """One per-variable sumcheck round's message: the round polynomial sent plus
    the Fiat-Shamir challenge it induced. `prove` stacks these over the scan, so
    the returned `msgs.round_poly` is the proof and `msgs.challenge` is the
    evaluation point. The challenge is re-derivable from `round_poly`, so it never
    goes on the wire -- it rides here only to spare the prover a transcript replay.
    A registered pytree because the `lax.scan` stacks it as its output;
    `LogupSumcheckRound.__call__` also emits one for the `fold_rounds` driver."""

    round_poly: Array
    challenge: Array


def fold_rounds(
    round: ProverRound, state: Any, transcript: Transcript, rounds: int
) -> tuple[Any, Transcript, list[Any]]:
    """Run `round` exactly `rounds` times; return (state, transcript, list[msg])."""
    msgs: list[Any] = []
    for _ in range(rounds):
        state, transcript, msg = round(state, transcript)
        msgs.append(msg)
    return state, transcript, msgs


class SumcheckSummand(Protocol):
    """The seam the homogeneous `prove` scan driver needs from a per-variable
    round: the round-poly `degree`, and `_combine` — the summand over the lifted
    factors. The driver owns the split / mask / fold / scan, so one scan serves
    every sumcheck; `sumcheck.prover.SumcheckRound` (product) and
    `logup_gkr.prover.LogupSumcheckRound` (LogUp) both satisfy it.

    `degree` is a read-only property here so a frozen-dataclass field (product)
    and a `@property` (LogUp) both match — a plain `degree: int` would demand a
    settable attribute that neither provides."""

    @property
    def degree(self) -> int: ...

    def _combine(self, *factors: Array) -> Array: ...


def prove(
    round: SumcheckSummand, state: Sequence[Array], transcript: Transcript
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Scan a sumcheck round once per variable; return the folded state, the
    advanced transcript, and the stacked per-round `RoundMsg` (`.round_poly` is the
    proof, `.challenge` is the evaluation point).

    Generic over the round's summand (`_combine`): the product and LogUp
    per-variable loops share this scan. The heterogeneous case (distinct rounds in
    sequence) stays on `fold_rounds`.
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
    ) -> tuple[tuple[list[Array], Transcript, Array], RoundMsg]:
        state, transcript, half = carry
        live = jnp.arange(half_max) < half
        pairs = [
            (buf[..., :half_max], lax.dynamic_slice_in_dim(buf, half, half_max, -1))
            for buf in state
        ]
        integrand = round._combine(
            *[lift_to_domain(p0, p1, degree) for p0, p1 in pairs]
        )
        msg = jnp.sum(
            jnp.where(live, integrand, jnp.zeros((), integrand.dtype)), axis=-1
        )
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = [
            jnp.concatenate([fold_pair(p0, p1, r[0]), jnp.zeros_like(p0)], axis=-1)
            for p0, p1 in pairs
        ]
        return (state, transcript, half // 2), RoundMsg(msg, r[0])

    init = (list(state), transcript, jnp.int32(half_max))
    (state, transcript, _), msgs = lax.scan(step, init, xs=None, length=rounds)
    return [buf[..., :1] for buf in state], transcript, msgs
