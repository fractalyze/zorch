# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover: per-variable rounds, the fold/lift helpers they share, and the
homogeneous scan driver `prove`.

A sumcheck round splits each MLE on the current variable, sends the round
polynomial over the domain [0..degree], then folds every MLE at the verifier's
challenge (P0 + r*(P1 - P0)). The split/validate and fold steps are summand-
independent, so they live as `split_halves` / `factors_on_domain` / `fold`
(plus the LSB stride-2 duals `split_pairs` / `fold_lsb` the jagged engines
bind by, and `zero_extend` — the fixed-width re-pad the scan carry below
pairs with its fold) and
each round supplies only its summand via `_round_poly`: `SumcheckRound` (here)
sums a product of factors; `LogupSumcheckRound` (in zorch.logup_gkr.prover) sums
the LogUp combine. The round body is element-wise field ops plus the one inherent
Sigma (no reduce/gather beyond it). The verifier dual lives in
`zorch.sumcheck.verifier`.

`prove` is the multilinear sumcheck driver, generic over the round's summand: the
per-variable loop becomes one `lax.scan` so the whole proof compiles to a single
traced region, flat in the variable count (issue #58) — an unrolled loop would
inflate the graph past the ZKX PTX cliff. The driver owns the split / mask / fold /
scan and reads only the round's `degree` + `_combine` (the `SumcheckSummand` seam),
so the product `SumcheckRound` and the LogUp `LogupSumcheckRound` share this one
scan. The catch is that a `scan` carry must keep a fixed shape, but the MLE state
halves every round; the carry holds each factor in a full-width buffer with the
live data packed at the front, masks the dead tail out of the round-poly sum, and
folds back into the front, zero-padding the rest — byte-identical to the
Python-loop fold. This is the multilinear-sumcheck specialization; the
scheme-agnostic round loop (any `Round`, any message shape) is
`zorch.prove.fold_rounds`. A univariate / FFT prover would be its own driver.

Fiat-Shamir stays a per-permute `zorch.poseidon2` marker (one fused kernel per
permute, fractalyze/xla_fork#76) and the reduce/fold is its own kernel — the path
a vendor compiles at scale. (An earlier `zorch.sumcheck` whole-scan megakernel
that wrapped the entire scan register-resident was dropped: it ptxas-overflowed
shared memory around 2^20, so it never compiled at real sizes.)
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.round import Round
from zorch.transcript import DuplexTranscript, Transcript, reinterpret_challenge
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.round import ProverRound


def split_halves(state: Sequence[Array]) -> list[tuple[Array, Array]]:
    """Validate, then halve each MLE on the current variable: [(P0, P1), ...].

    Factors must be non-empty, share a shape, and have an even width -- fail
    loud rather than silently drop the odd element on `// 2`."""
    if not state:
        raise ValueError("state must hold at least one factor")
    shape = state[0].shape
    out = []
    for i, evals in enumerate(state):
        if evals.shape != shape:
            raise ValueError(
                f"all factors must share a shape; factor {i} is {evals.shape}, "
                f"factor 0 is {shape}"
            )
        if evals.shape[-1] % 2 != 0:
            raise ValueError(f"factor width must be even, got {evals.shape[-1]}")
        half = evals.shape[-1] // 2
        out.append((evals[..., :half], evals[..., half:]))
    return out


def fold_pair(p0: Array, p1: Array, r: Array) -> Array:
    """Fold one split pair at challenge `r`: P0 + r*(P1 - P0)."""
    return p0 + r * (p1 - p0)


def split_pairs(arr: Array) -> tuple[Array, Array]:
    """Split on the LSB variable: the stride-2 `(arr[..., 0::2], arr[..., 1::2])`
    consecutive-pair dual of `split_halves`' contiguous MSB halves. The jagged
    engines bind LSB-first -- a batch-major jagged layout makes the row LSB the
    in-segment pair dimension, so the pair fold never crosses a segment
    boundary -- while the dense drivers here stay MSB-halving."""
    return arr[..., 0::2], arr[..., 1::2]


def fold_lsb(arr: Array, r: Array) -> Array:
    """Bind the LSB variable at challenge `r`: `fold_pair` over the stride-2
    pairs. Halves the last axis; the LSB dual of `fold`."""
    p0, p1 = split_pairs(arr)
    return fold_pair(p0, p1, r)


def zero_extend(arr: Array, width: int) -> Array:
    """Zero-extend the last axis to `width` -- the re-extension a fixed-shape
    round carry pairs with a fold: the live prefix halves each round while the
    buffer width stays put, and the dead tail stays exactly zero so full-width
    reductions match live-prefix-truncated ones byte-for-byte (field zero-adds
    are exact)."""
    pad = width - arr.shape[-1]
    if pad < 0:
        raise ValueError(f"width {width} < last-axis size {arr.shape[-1]}")
    if pad == 0:
        return arr
    return jnp.concatenate([arr, jnp.zeros((*arr.shape[:-1], pad), arr.dtype)], axis=-1)


def lift_to_domain(p0: Array, p1: Array, degree: int, start: int = 0) -> Array:
    """Lift one split pair to the evaluation domain [start..degree]:
    f[u] = P0 + u*(P1 - P0), shape (degree+1-start, *P0.shape).

    `start` defaults to 0 (the natural domain [0..degree]); set it to 1 to omit
    the f[0] = P0 point, the compressed round-poly wire form a verifier
    reconstructs from the running claim (s(0) = claim - s(1)). The whole u-domain
    is built at once so the round poly stays one batched reduction (not
    degree+1 separate ones). `us` uses jnp.stack (not jnp.arange, whose iota is
    unsupported for extension dtypes) and is reshaped to broadcast over any
    leading batch dims of the factor."""
    us = jnp.stack([jnp.array(u, p0.dtype) for u in range(start, degree + 1)])
    return p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0)


def fold(state: Sequence[Array], r: Array) -> list[Array]:
    """Fold each MLE at challenge `r`: P0 + r*(P1 - P0). Halves width."""
    return [fold_pair(p0, p1, r) for (p0, p1) in split_halves(state)]


def factors_on_domain(state: Sequence[Array], degree: int) -> list[Array]:
    """Lift each split factor to the round's evaluation domain [0..degree]:
    f_k[u] = P0_k + u*(P1_k - P0_k), one array of shape (degree+1, *batch) each."""
    return [lift_to_domain(p0, p1, degree) for (p0, p1) in split_halves(state)]


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Product sumcheck: s = sum_x prod_k P_k(x), one factor per state entry."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def combine_scalars(self) -> tuple[Array, ...]:
        """No loop-invariant scalars: the product summand reads only its factors."""
        return ()

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        """Product summand `prod_k f_k` (the scalar-explicit seam; product takes no
        scalars). Single source of the combine math: `_combine` and the round-poly
        reduction both route here, so they cannot drift."""
        del scalars  # product has none
        return reduce(operator.mul, factors)

    def _combine(self, *factors: Array) -> Array:
        """Product summand bound to its (empty) scalars; the scan driver reads only
        this, so the round owns its summand and the driver stays summand-generic."""
        return self.combine(self.combine_scalars(), *factors)

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)), shape (degree+1, *batch).

        One batched reduction over the whole u-domain, so it lowers toward a
        single reduction kernel rather than degree+1 separate ones."""
        return jnp.sum(self._combine(*factors_on_domain(state, self.degree)), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, Array]:
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = fold(state, r[0])
        return state, transcript, msg


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
@dataclass(frozen=True)
class CompressedProductRound(Round):
    """Two-factor product round with the compressed coefficient wire: the message
    is `[c_0, c_2]` — the degree-2 round polynomial's constant and leading
    coefficients — and the linear coefficient stays off the wire (the verifier
    dual, `verifier.CompressedCoeffsSumcheckRound`, reconstructs it from the
    running claim via `s(0) + s(1) = claim`). Split and fold match
    `SumcheckRound(degree=2)` exactly (the MSB variable binds); only the message
    form differs, so a scheme whose wire fixes this form swaps rounds without
    touching the fold. `c_2 = Σ (P1_f - P0_f)·(P1_b - P0_b)` is the honest
    leading coefficient in any characteristic; over char 2 it coincides with the
    `(P0 + P1)` products some wire specs write it as."""

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """`[c_0, c_2]` of `s(X) = Σ_x' f(X, x')·b(X, x')`, shape (2, *batch):
        one stacked element-wise product per coefficient, then the single
        inherent Σ."""
        if len(state) != 2:
            raise ValueError(
                f"compressed product round takes exactly 2 factors, got {len(state)}"
            )
        (f0, f1), (b0, b1) = split_halves(state)
        return jnp.sum(jnp.stack([f0 * b0, (f1 - f0) * (b1 - b0)]), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, Array]:
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = fold(state, r[0])
        return state, transcript, msg


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["round_poly", "challenge"],
    meta_fields=[],
)
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


class SumcheckSummand(Protocol):
    """The seam the homogeneous `prove` scan driver needs from a per-variable
    round: the round-poly `degree`, and `_combine` — the summand over the lifted
    factors. The driver owns the split / mask / fold / scan, so one scan serves
    every sumcheck; `SumcheckRound` (product) and
    `logup_gkr.prover.LogupSumcheckRound` (LogUp) both satisfy it.

    `degree` is a read-only property here so a frozen-dataclass field (product)
    and a `@property` (LogUp) both match — a plain `degree: int` would demand a
    settable attribute that neither provides.

    `combine` is the scalar-explicit form of the summand and `combine_scalars` the
    loop-invariant scalars it reads (LogUp's λ; empty for product), hoisted out of
    the per-variable scan so they bind once rather than per round."""

    @property
    def degree(self) -> int: ...

    def combine_scalars(self) -> tuple[Array, ...]: ...

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array: ...

    def _combine(self, *factors: Array) -> Array: ...


# The FS-less compute-only round marker (zorch#327): the jagged LogUp-GKR host
# loop wraps each round's fold+sum (no Fiat-Shamir) in this composite, while the
# separate `zorch.poseidon2` marker carries FS between rounds. The composite
# attributes the recognizing emitter parses:
#   phase   -- "first" (round 0, no fold), "mid" (fold-by-alpha then sum),
#              "boundary" (the row->interaction handoff: fold-by-alpha then sum
#              over the still-unfolded eq, which rides through un-bound), or
#              "final" (fold only, emitting the four pair openings); routes to
#              the round kernel by position.
#   variant -- the round-kernel shape: "dense" (the uniform interaction round --
#              a batched LogUp-GKR round-poly, folds densely) or "jagged" (the row
#              phase -- segment-based, variable heights, runtime row_counts).
#   degree    -- round-poly degree.
#   poly_form -- the round-poly evaluation domain/form ("coefficient" = the Gruen
#                {0, 1/2} + s(1)=claim-s(0) scheme this round uses).
# No `num_scalars`: the LogUp summand is hardcoded in the emitter (AccumLogupPair),
# which does not read it (it matters only for a generic `zorch.sumcheck.combine`
# region). No `challenge_limbs`: the separate FS step recomposes the fold
# challenge, which arrives as one operand whose dtype already carries base vs
# extension.
SUMCHECK_ROUND_MARKER = "zorch.sumcheck.round"
# Version 1: this marker never shipped, and its producer (here), the zkx
# `SumcheckRecognizer` (`kSumcheckRoundCompositeVersion`), and the emitters are
# pinned together, so the version is the initial one and moves only on a future
# cross-release ABI break. Keep in lockstep with the zkx recognizer's constant.
SUMCHECK_ROUND_MARKER_VERSION = 1


def prove(
    round: SumcheckSummand,
    state: Sequence[Array],
    transcript: Transcript,
    *,
    eval_start: int = 0,
    challenge_dtype: object | None = None,
    challenge_limbs: int = 1,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Scan a sumcheck round once per variable; return the folded state, the
    advanced transcript, and the stacked per-round `RoundMsg` (`.round_poly` is the
    proof, `.challenge` is the evaluation point).

    Generic over the round's summand (`_combine`): the product and LogUp
    per-variable loops share this scan. The heterogeneous case (distinct rounds in
    sequence) stays on `zorch.prove.fold_rounds`.

    `eval_start` controls the round-poly evaluation domain `[eval_start..degree]`:
    the default 0 sends `degree+1` values; 1 omits `s(0)` and sends `degree`, the
    compressed wire form a verifier reconstructs as `s(0) = claim − s(1)`.
    `challenge_dtype` / `challenge_limbs` pick the per-round fold challenge's
    field: the default (None / 1) folds with one transcript-field squeeze; a
    challenge in an extension reinterprets `challenge_limbs` consecutive squeezes
    as one `challenge_dtype` element (the `sample_challenge` packing, applied
    inside the scan), as a downstream scheme's extension-field sumchecks need.
    """
    state = list(state)
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    width = state[0].shape[-1]
    if log2_strict_usize(width) == 0:
        raise ValueError(
            f"prove requires a state width >= 2 (at least one round), got width {width}"
        )
    if not 0 <= eval_start <= round.degree:
        raise ValueError(
            f"eval_start must be within [0, degree {round.degree}], got {eval_start}"
        )
    if challenge_limbs < 1:
        raise ValueError(f"challenge_limbs must be >= 1, got {challenge_limbs}")
    if challenge_dtype is None and challenge_limbs != 1:
        # The default squeeze is the identity reinterpret (one transcript-field
        # element). >1 limbs with no dtype to pack them into would advance the
        # transcript past squeezes the fold never consumes — a silent verifier
        # desync; reject it. (The dtype != None case is guarded at the squeeze.)
        raise ValueError(
            "challenge_limbs must be 1 when challenge_dtype is None, got "
            f"{challenge_limbs}"
        )
    if isinstance(transcript, DuplexTranscript) and transcript.fs_on_host:
        # The host sponge is an eager primitive (`device_put` / `.devices()` on its
        # inputs); it cannot run inside `_prove_scan`'s `lax.scan` body. A host-FS
        # transcript must drive the host-relaunch round engine
        # (`logup_gkr.jagged_prover`), not this dense scan prove — fail loud rather
        # than abort deep in the scan trace with a ConcretizationError.
        raise NotImplementedError(
            "fs_on_host is unsupported on the dense sumcheck scan path; drive the "
            "host-relaunch round engine instead"
        )
    return _prove_scan(
        round,
        state,
        transcript,
        eval_start=eval_start,
        challenge_dtype=challenge_dtype,
        challenge_limbs=challenge_limbs,
    )


def _prove_scan(
    round: SumcheckSummand,
    state: list[Array],
    transcript: Transcript,
    *,
    eval_start: int = 0,
    challenge_dtype: object | None = None,
    challenge_limbs: int = 1,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """The per-variable sumcheck scan: split / lift / round-poly / Fiat-Shamir /
    fold, one `lax.scan` step per variable. The body `prove` always runs, on a
    validated, non-empty `state` (`prove` guards the empty / zero-round cases).

    `eval_start` / `challenge_dtype` / `challenge_limbs` are the round-poly domain
    and fold-challenge-field controls `prove` documents."""
    width = state[0].shape[-1]
    rounds = log2_strict_usize(width)
    degree = round.degree
    half_max = width // 2
    scalars = round.combine_scalars()

    def step(
        carry: tuple[list[Array], Transcript, Array], _: None
    ) -> tuple[tuple[list[Array], Transcript, Array], RoundMsg]:
        state, transcript, half = carry
        live = jnp.arange(half_max) < half
        pairs = [
            (buf[..., :half_max], lax.dynamic_slice_in_dim(buf, half, half_max, -1))
            for buf in state
        ]
        lifted = [lift_to_domain(p0, p1, degree, eval_start) for p0, p1 in pairs]
        integrand = round.combine(scalars, *lifted)
        msg = jnp.sum(
            jnp.where(live, integrand, jnp.zeros((), integrand.dtype)), axis=-1
        )
        # One round challenge: `challenge_limbs` squeezes reinterpreted as a single
        # `challenge_dtype` element (the `sample_challenge` packing) — the identity
        # squeeze at the defaults, an extension-field challenge otherwise.
        transcript, raw = transcript.observe_and_sample(msg, challenge_limbs)
        r = (
            raw[0]
            if challenge_dtype is None
            else reinterpret_challenge(raw, challenge_dtype)
        )
        state = [zero_extend(fold_pair(p0, p1, r), width) for p0, p1 in pairs]
        return (state, transcript, half // 2), RoundMsg(msg, r)

    init = (state, transcript, jnp.int32(half_max))
    (state, transcript, _), msgs = lax.scan(step, init, xs=None, length=rounds)
    return [buf[..., :1] for buf in state], transcript, msgs


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _summand: type[SumcheckSummand] = SumcheckRound
    _prover_round: type[ProverRound] = SumcheckRound
    _compressed_prover_round: type[ProverRound] = CompressedProductRound
