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

`prove_fs` is the register-resident sibling: it wraps `prove`'s scan — unchanged,
Fiat-Shamir already inside it — in a hash-agnostic `zorch.sumcheck` composite a
vendor codegens as one on-chip kernel. The duplex sponge threads through as
operands and the FS permutation rides as a nested `poseidon2:` marker (round
constants auto-lift), so the marker names no hash; unrecognized it inlines
bit-equal to `prove`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import register_dataclass

from zorch.fusion import fused_region
from zorch.round import Round
from zorch.sumcheck.prover import fold_pair, lift_to_domain
from zorch.transcript import DuplexState, DuplexTranscript, Transcript
from zorch.utils.bits import log2_strict_usize

SUMCHECK_MARKER = "zorch.sumcheck"


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
    round: Round, state: Any, transcript: Transcript, rounds: int
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


def prove_fs(
    round: SumcheckSummand,
    state: Sequence[Array],
    transcript: DuplexTranscript,
) -> tuple[Array, DuplexTranscript]:
    """Wrap `prove`'s scan in a `zorch.sumcheck` composite, Fiat-Shamir INSIDE.

    The register-resident sibling of `prove`: the body is `prove` itself, so the
    `lax.scan` (FS sampled in-step) and the proof are bit-identical — no second
    prover, no unrolled copy. A vendor codegens the marker as one on-chip kernel
    (state never round-trips HBM between rounds); unrecognized, the composite
    inlines to the same `(proof, transcript)` `prove` produces.

    The duplex sponge threads through the marker as its five state leaves (the
    mutable carry), and the FS permutation rides as a *nested* `poseidon2:` marker
    whose round constants auto-lift into this composite's operands — so the marker
    carries no hash identity (the vendor reads it from the nested marker) and no
    pre-sampled challenges. Operand ABI: `[factor tables][transcript leaves]` plus
    the auto-lifted round constants; results are `[proof][transcript leaves]`. The
    proof is the flat round-major `[num_vars*(degree+1)]`; `degree`/`num_vars` ride
    in the name, the only sumcheck structure the emitter needs.
    """
    if not state:
        raise ValueError("prove_fs requires a non-empty state (one Array per factor)")
    num_vars = log2_strict_usize(state[0].shape[-1])
    if num_vars == 0:
        raise ValueError("prove_fs requires a state width >= 2 (at least one round)")
    perm = transcript.permutation
    rate = transcript.rate
    num_factors = len(state)
    st = transcript.state
    leaves = (
        st.input_buffer,
        st.output_buffer,
        st.sponge_state,
        st.in_pos,
        st.out_pos,
    )
    n_leaves = len(leaves)

    def body(*operands: Array) -> tuple[Array, ...]:
        tables = list(operands[:num_factors])
        lv = operands[num_factors : num_factors + n_leaves]
        # Rebuild the transcript from its leaves so the body closes over no sponge
        # state; `prove`'s scan does the per-round FS, so this stays the one prover.
        _, t, msgs = prove(
            round, tables, DuplexTranscript(perm, rate, DuplexState(*lv))
        )
        # prove threads the same DuplexTranscript back, but it is typed as the
        # generic `Transcript` protocol; recover the sponge leaves for the result.
        s = cast(DuplexTranscript, t).state
        return (
            msgs.round_poly.reshape(-1),  # round-major [num_vars*(degree+1)]
            s.input_buffer,
            s.output_buffer,
            s.sponge_state,
            s.in_pos,
            s.out_pos,
        )

    proof, *out_leaves = fused_region(
        body, *state, *leaves, name=f"{SUMCHECK_MARKER}:{round.degree}:{num_vars}"
    )
    return proof, DuplexTranscript(perm, rate, DuplexState(*out_leaves))
