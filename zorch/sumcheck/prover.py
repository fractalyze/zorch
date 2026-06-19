# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover: per-variable rounds, the fold/lift helpers they share, and the
homogeneous scan driver `prove` (with the register-resident `zorch.sumcheck`
marker).

A sumcheck round splits each MLE on the current variable, sends the round
polynomial over the domain [0..degree], then folds every MLE at the verifier's
challenge (P0 + r*(P1 - P0)). The split/validate and fold steps are summand-
independent, so they live as `split_halves` / `factors_on_domain` / `fold` and
each round supplies only its summand via `_round_poly`: `SumcheckRound` (here)
sums a product of factors; `LogupSumcheckRound` (in zorch.logup_gkr.prover) sums
the LogUp combine. The round body is element-wise field ops plus the one inherent
Sigma (no reduce/gather beyond it), so each round stays wrappable by a single-
kernel marker without restructuring. The verifier dual lives in
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

When the transcript's Fiat-Shamir permutation lowers to a dedicated fusion marker
(`transcript.has_dedicated_fusion`, the `zorch.merkle_commit` gate), `prove` wraps
that whole scan in one hash-agnostic `zorch.sumcheck` composite a vendor codegens
as a single register-resident kernel — reduce → Fiat-Shamir → fold looped on-chip,
the MLE state and the sponge never round-tripping HBM between rounds. The marker
is transparent (callers still just call `prove`) and carries computation, not
parameters: no pre-sampled challenges (FS stays inside the body scan), no hash
identity (it rides as a nested `zorch.poseidon2` marker whose round constants auto-lift,
exactly as `merkle_commit` keeps the hash off its marker). Unrecognized — a vendor
without the emitter, or a jaxlib without `stablehlo.CompositeOp` — it decomposes to
the same scan, byte-identical and sound. A test `CheapPermutation`
(`has_dedicated_fusion=False`) keeps the plain scan, so unit tests never need the
marker. This is the register-resident lever reached transparently from `prove`.
"""
from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.fusion import fused_region
from zorch.round import Round
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    Transcript,
    reinterpret_challenge,
)
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
        scalars). Single source of the combine math: `_combine`, the round-poly
        reduction, and the marked path's nested `zorch.sumcheck.combine` region all
        route here, so they cannot drift."""
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
    loop-invariant scalars it reads (LogUp's λ; empty for product). The marked path
    threads those scalars as marker operands and delineates `combine` as a nested
    `zorch.sumcheck.combine` region, so a vendor inlines any summand generically."""

    @property
    def degree(self) -> int: ...

    def combine_scalars(self) -> tuple[Array, ...]: ...

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array: ...

    def _combine(self, *factors: Array) -> Array: ...


# Hash-agnostic boundary `prove` wraps its whole scan in when the transcript's FS
# permutation has a dedicated fusion marker. The structural shape rides in
# `composite.attributes` (`degree`/`num_vars`/`num_factors`), not the name — named,
# typed, and optional-extensible (the recognizer's contract, vs a brittle
# positional name suffix). The combine is a body sub-region and the hash rides as a
# nested `zorch.poseidon2` marker (round constants auto-lift), so the marker names no
# hash and carries no pre-sampled challenges.
SUMCHECK_MARKER = "zorch.sumcheck"
# Marker revision the zkx `SumcheckRecognizer` gates on (`composite.version`).
SUMCHECK_MARKER_VERSION = 1

# Nested region delineating the per-round combine (the summand over the lifted
# factors) inside the sumcheck scan, so a vendor inlines it generically rather
# than hardcoding one combine (#113). Carries the computation, not
# params: operands are `[lifted factors][combine scalars]` and the body is
# `round.combine`; found by recursive search like the nested `zorch.poseidon2` marker.
SUMCHECK_COMBINE_MARKER = "zorch.sumcheck.combine"
SUMCHECK_COMBINE_MARKER_VERSION = 1


def prove(
    round: SumcheckSummand,
    state: Sequence[Array],
    transcript: Transcript,
    *,
    num_real: int | None = None,
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

    When the transcript's Fiat-Shamir permutation lowers to a dedicated fusion
    marker (`has_dedicated_fusion`), the whole scan is wrapped in one
    `zorch.sumcheck` composite a vendor codegens register-resident (see the module
    docstring); otherwise it runs directly. The two are byte-identical — the marker
    decomposes to the same scan — so callers see one `prove`.

    `num_real` declares how many leading entries of each factor table are real,
    the rest being zero padding (a jagged table padded to a power of two). It
    rides on the marker as an optional composite attribute so a vendor bounds the
    first round's reduction to the real prefix; the computation is unchanged —
    skipping the tail is sound only because a padded summand sums to zero there,
    which is the caller's contract.

    `eval_start` controls the round-poly evaluation domain `[eval_start..degree]`:
    the default 0 sends `degree+1` values; 1 omits `s(0)` and sends `degree`, the
    compressed wire form a verifier reconstructs as `s(0) = claim − s(1)`.
    `challenge_dtype` / `challenge_limbs` pick the per-round fold challenge's
    field: the default (None / 1) folds with one transcript-field squeeze; a
    challenge in an extension reinterprets `challenge_limbs` consecutive squeezes
    as one `challenge_dtype` element (the `sample_challenge` packing, applied
    inside the scan), as the SWIRL stacking / zerocheck sumchecks need. A
    non-default `eval_start` or challenge field bypasses the dedicated-fusion
    marker — it codegens only the default round shape, so those decompose to the
    plain scan (byte-identical); a marker for them is a follow-up.
    """
    state = list(state)
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    width = state[0].shape[-1]
    if log2_strict_usize(width) == 0:
        raise ValueError(
            "prove requires a state width >= 2 (at least one round), got "
            f"width {width}"
        )
    # Mirror the zkx recognizer's bound so a bad value fails at emission, and on
    # the unmarked path too — the two paths must reject identically.
    if num_real is not None and not 1 <= num_real <= width:
        raise ValueError(
            f"num_real must be within [1, table width {width}], got {num_real}"
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
    # Mark only a DuplexTranscript with a dedicated-fusion permutation, AND only
    # the default round shape — the marker threads the sponge's five-leaf state as
    # operands (so it needs that concrete layout) and codegens a full-domain,
    # single-transcript-field-challenge round, so a truncated domain or an
    # extension challenge falls through to the plain scan. The isinstance narrows
    # the type for `_prove_marked` (the merkle.py `has_dedicated_fusion` gate).
    default_round = eval_start == 0 and challenge_dtype is None and challenge_limbs == 1
    if (
        default_round
        and isinstance(transcript, DuplexTranscript)
        and transcript.has_dedicated_fusion
    ):
        return _prove_marked(round, state, transcript, num_real=num_real)
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
    mark_combine: bool = False,
    scalars: Sequence[Array] | None = None,
    eval_start: int = 0,
    challenge_dtype: object | None = None,
    challenge_limbs: int = 1,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """The per-variable sumcheck scan: split / lift / round-poly / Fiat-Shamir /
    fold, one `lax.scan` step per variable. This is both the plain prover body and
    the decomposition `prove`'s `zorch.sumcheck` marker wraps, so the marked and
    unmarked paths are byte-identical by construction. Assumes a validated,
    non-empty `state` (`prove` guards the empty / zero-round cases).

    `eval_start` / `challenge_dtype` / `challenge_limbs` are the round-poly domain
    and fold-challenge-field controls `prove` documents; the marked path leaves
    them at their defaults (full domain, single transcript-field challenge).

    `mark_combine` wraps each round's combine in a nested `zorch.sumcheck.combine`
    region (#113) — set only on the marked path, where `_prove_marked`
    threads the combine `scalars` (λ) in as marker operands so the body reads them
    instead of closing over the round (no duplicate auto-lift). The region is
    transparent, so the wrapped scan stays byte-identical to the plain one. When
    `scalars is None` the round binds its own (the plain path)."""
    width = state[0].shape[-1]
    rounds = log2_strict_usize(width)
    degree = round.degree
    half_max = width // 2
    n_factors = len(state)
    scalars = round.combine_scalars() if scalars is None else scalars

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
        if mark_combine:
            # operands [lifted factors][scalars]; body reads passed scalars, so the
            # round (and its λ) is not captured — λ rides as an explicit operand.
            integrand = fused_region(
                lambda *ops: round.combine(ops[n_factors:], *ops[:n_factors]),
                *lifted,
                *scalars,
                name=SUMCHECK_COMBINE_MARKER,
                version=SUMCHECK_COMBINE_MARKER_VERSION,
            )
        else:
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
        state = [
            jnp.concatenate([fold_pair(p0, p1, r), jnp.zeros_like(p0)], axis=-1)
            for p0, p1 in pairs
        ]
        return (state, transcript, half // 2), RoundMsg(msg, r)

    init = (state, transcript, jnp.int32(half_max))
    (state, transcript, _), msgs = lax.scan(step, init, xs=None, length=rounds)
    return [buf[..., :1] for buf in state], transcript, msgs


def _state_leaves(st: DuplexState) -> tuple[Array, Array, Array, Array, Array]:
    """The five `DuplexState` arrays in field order — the marker threads them as its
    mutable transcript operands and reads them back as results, so the producer here
    and the zkx consumer share this one ordering."""
    return (st.input_buffer, st.output_buffer, st.sponge_state, st.in_pos, st.out_pos)


def _prove_marked(
    round: SumcheckSummand,
    state: list[Array],
    transcript: DuplexTranscript,
    num_real: int | None = None,
) -> tuple[list[Array], DuplexTranscript, RoundMsg]:
    """Wrap `_prove_scan` in the hash-agnostic `zorch.sumcheck` composite —
    Fiat-Shamir INSIDE, so the body is the *same* scan and the folded state,
    advanced transcript, and proof are bit-identical to the plain path.

    The shape (`degree`, `num_vars`, `num_factors`, plus `num_real` when given)
    rides as `composite.attributes` under `version` 1 — the recognizer's
    contract; the body ignores them (metadata only). The duplex sponge threads
    through as the five `DuplexState` leaves (the
    mutable carry); the FS permutation rides as the nested `zorch.poseidon2` marker
    inside `observe_and_sample`, whose round constants auto-lift into this
    composite's operands — so the marker names no hash and carries no pre-sampled
    challenges. Operands: `[factor tables][combine scalars][5 sponge leaves]` plus
    the auto-lifted round constants; results: `[folded state][5 sponge leaves]
    [round polys][challenges]` — `prove`'s return, repacked. The per-round combine
    rides as a nested `zorch.sumcheck.combine` region and its scalars (λ) as the
    `[combine scalars]` operand segment — empty for product, so a product marker's
    operand layout is unchanged.
    """
    perm, rate = transcript.permutation, transcript.rate
    num_vars = log2_strict_usize(state[0].shape[-1])
    n_factors = len(state)
    scalars = round.combine_scalars()
    n_scalars = len(scalars)
    leaves = _state_leaves(transcript.state)

    def body(
        *operands: Array, **_attrs: object
    ) -> tuple[list[Array], tuple[Array, Array, Array, Array, Array], RoundMsg]:
        tables = list(operands[:n_factors])
        sca = operands[n_factors : n_factors + n_scalars]
        lv = operands[n_factors + n_scalars : n_factors + n_scalars + len(leaves)]
        # Rebuild the transcript from its leaves so the body closes over no sponge
        # state; `_prove_scan` runs the per-round FS, keeping this the one prover.
        # Forward the combine scalars so the nested combine region reads them as
        # operands. `_attrs` is marker metadata the inline fallback passes through
        # — the decomposition itself does not read it.
        folded, t, msgs = _prove_scan(
            round,
            tables,
            DuplexTranscript(perm, rate, DuplexState(*lv)),
            mark_combine=True,
            scalars=sca,
        )
        return folded, _state_leaves(cast(DuplexTranscript, t).state), msgs

    # Optional attr: emitted only when declared, so a dense prove's envelope is
    # unchanged for a recognizer that predates `num_real`.
    opt_attrs = {} if num_real is None else {"num_real": num_real}
    folded, out_leaves, msgs = fused_region(
        body,
        *state,
        *scalars,
        *leaves,
        name=SUMCHECK_MARKER,
        version=SUMCHECK_MARKER_VERSION,
        degree=round.degree,
        num_vars=num_vars,
        num_factors=n_factors,
        **opt_attrs,
    )
    return folded, DuplexTranscript(perm, rate, DuplexState(*out_leaves)), msgs


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _summand: type[SumcheckSummand] = SumcheckRound
    _prover_round: type[ProverRound] = SumcheckRound
