# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged LogUp-GKR prover: the materialized per-layer sumcheck.

The jagged sibling of `prover.GkrLayerRound`. A `JaggedGkrLayer` materializes
only `sum(row_counts)` of its virtual `2^(niv+nrv)` positions; every
non-materialized position holds the fold-neutral fraction (n=0, d=1), whose
LogUp summand `eq * (lam*(n0*d1 + n1*d0) + d0*d1)` collapses to just its eq
weight. The sumcheck therefore runs over the materialized arrays and adds the
virtual mass back in closed form: the eq weights of a full hypercube sum to
the product of the bound variables' eq factors (`pad_adj`), so the
correction per round is `pad_adj - eq_sum_materialized`.

Round polynomials travel in coefficient form, interpolated through
{0, 1, 1/2, b}: the summand carries the current variable's eq factor, whose
root `b = (1-z)/(1-2z)` is known to both sides, so a degree-3 round needs
only the materialized evaluations at {0, 1/2} plus `s(1) = claim - s(0)`
(Gruen, https://eprint.iacr.org/2024/108). Value-form on the natural domain
would need a third materialized evaluation per round.

Variables bind LSB-first (consecutive-pair fold): a jagged layer is
batch-major, so the row LSB is the in-segment pair dimension and the
stride-2 fold never crosses a segment boundary once odd segments are
re-padded (the same `_segment_gather` machinery as the circuit transition).
Row variables fold first while their eq factor rides as the materialized
`eq_row` lookup; once rows are exhausted the accumulated row-eq residual
becomes the scalar `eq_adj` and the batch variables fold densely. The
bound point is challenges reversed -- LSB-first binding makes the last
challenge the MSB -- so the carry convention (MSB-first point, child selector
appended last) matches the dense chain's.

Per-round shapes shrink and the gather layout changes round to round, so the
driver is a host-orchestrated Python loop over plain numeric bodies, not the
homogeneous `zorch.sumcheck` scan (see docs/conventions.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, export
from jax._src.export._export import call_exported_p as _call_exported_p

from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _pad_neutral,
    _prepad_folded,
    _segment_gather,
    _segment_gather_np,
)
from zorch.logup_gkr.prover import Carry, LogupSummand, fold_carry
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
)
from zorch.round import Round
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import (
    DuplexTranscript,
    Transcript,
    observe_and_sample_marked,
    reinterpret_challenge,
    sample_challenge,
)

if TYPE_CHECKING:
    from zorch.round import ProverRound

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2), in coefficient form.
_DEGREE = 3


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "lam",
        "claim",
        "round_polys",
        "point",
        "numerator_0",
        "numerator_1",
        "denominator_0",
        "denominator_1",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class JaggedLayerProof:
    """One jagged GKR layer's sumcheck transcript: the batching challenge and
    opening claim the layer entered with (the per-layer anchors a consumer
    diffs first when a byte-match diverges mid-pyramid), the coefficient-form
    round polynomials, the bound point, and the final pair openings.

    A pytree (every field is an `Array`, like the dense `sumcheck.RoundMsg`) so
    it can be returned across a `jax.jit` boundary -- the per-layer jit the
    chained prover wraps each round in.

    `point` is retained for wire serialization despite being replay-derivable
    — `LayerProof.point` carries the rationale and the
    verifier-must-never-read rule."""

    lam: Array
    claim: Array
    round_polys: Array  # (num_variables, _DEGREE + 1), ascending coefficients
    point: Array  # the bound point, MSB-first (the sampled challenges reversed)
    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array


@cache
def _round_metadata(
    row_counts: tuple[int, ...], num_row_vars: int
) -> list[tuple[Array | None, Array, Array]]:
    """Per-round `(gather, col_index, pair_index)` for the row-variable phase.

    Memoized on the (static) layout: the schedule is a pure function of the
    Python-int row counts, so a cold trace reuses the device-resident index arrays
    across same-shape layers instead of rebuilding them (host numpy + a batched
    device_put). The arrays are tiny and immutable, so caching costs negligible
    device memory and cannot alias across the one-live-layer plane release.

    Round k folds the layout round k-1 left behind: odd segments pre-pad to
    even (`gather`; None when already even), then the stride-2 fold halves
    every segment. `col_index` maps each pair to its batch element and
    `pair_index` to its in-segment pair offset -- the eq_row lookup is
    segment-local because a jagged layer is batch-major, so the row-eq
    factor is indexed per segment while `eq_int[col_index]` carries the
    batch weight. All static, derived from the Python-int row counts.
    """
    # Build the whole schedule on the host, then commit it in ONE batched
    # device_put (not per round): every index array is tiny and static. The None
    # gathers (no re-pad) ride through device_put as empty pytree nodes.
    host_meta: list[tuple[np.ndarray | None, np.ndarray, np.ndarray]] = []
    counts = row_counts
    for _ in range(num_row_vars):
        padded, pairs = _prepad_folded(
            counts
        )  # the circuit's own prepad/fold recurrence
        col_index = np.repeat(np.arange(len(pairs), dtype=np.int32), pairs)
        pair_index = np.concatenate([np.arange(pc, dtype=np.int32) for pc in pairs])
        host_meta.append((_segment_gather_np(counts, padded), col_index, pair_index))
        counts = pairs
    # `ensure_compile_time_eval` forces the device_put to materialize a CONCRETE
    # committed array: this memoized builder is hit inside the round-zone trace, and
    # without it the cached value is a trace-scoped `device_put` tracer that escapes
    # when a later call reuses the cache (UnexpectedTracerError). Concrete -> the jit
    # bakes it as a constant.
    with jax.ensure_compile_time_eval():
        return jax.device_put(host_meta)


def _bind_lsb(arr: Array, r: Array) -> Array:
    """Bind the LSB variable: stride-2 consecutive pairs fold via the shared
    `sumcheck.prover.fold_pair` -- `e0 + r*(e1 - e0)`. (The split is LSB/stride-2,
    distinct from `fold`/`split_halves`' contiguous MSB halves; only the scalar
    fold is shared.)"""
    return fold_pair(arr[0::2], arr[1::2], r)


def _virtual_mass_correction(pad_adj: Array, eq_sum: Array) -> Array:
    """The virtual (non-materialized) mass a round adds back in closed form.

    The sumcheck runs only over the materialized rows; every non-materialized
    position holds the fold-neutral fraction (n=0, d=1), whose LogUp summand
    collapses to just its eq weight. The eq weights of the full remaining
    hypercube sum to `pad_adj`, so the virtual positions contribute exactly
    `pad_adj - eq_sum` (the full mass minus the materialized `eq_sum`) -- added
    back by `_round_coeffs` in closed form rather than iterated over."""
    return pad_adj - eq_sum


def _round_coeffs(
    eval_zero: Array,
    eval_half: Array,
    eq_sum: Array,
    eq_adj: Array,
    pad_adj: Array,
    z_cur: Array,
    claim: Array,
    naturals: Array,
    inv_vand: Array,
) -> Array:
    """Round polynomial coefficients from the materialized {0, 1/2} sums.

    Every non-materialized position contributes exactly its eq weight, and
    the eq weights of the full remaining hypercube sum to `pad_adj`, so the
    virtual mass is `pad_adj - eq_sum`: at u=0 it carries the current
    variable's eq factor `(1 - z_cur)`; at the doubled u=1/2 scale each
    virtual pair contributes `den_h = 4` at weight `eq_h = eq_rest`, and the
    doubled products overcount s(1/2) by 8. `eq_adj` is the row-eq residual
    scalar once the row variables are exhausted (1 before that).

    The interpolant through {0, 1, 1/2, b} crosses to coefficients via the
    natural domain: Lagrange-evaluate it at `naturals` ({0..3}), then
    `inv_vand` maps values to coefficients. Both are hoisted by the caller
    -- they only depend on the degree.
    """
    dtype = claim.dtype
    one = jnp.ones((), dtype)
    correction = _virtual_mass_correction(pad_adj, eq_sum)
    s_zero = (eval_zero + correction * (one - z_cur)) * eq_adj
    s_half = (
        (eval_half + correction * jnp.array(4, dtype)) / jnp.array(8, dtype) * eq_adj
    )
    s_one = claim - s_zero
    b_root = (one - z_cur) / (one - jnp.array(2, dtype) * z_cur)
    xs = jnp.stack([jnp.zeros((), dtype), one, one / jnp.array(2, dtype), b_root])
    ys = jnp.stack([s_zero, s_one, s_half, jnp.zeros((), dtype)])
    lagrange = jax.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, xs)
    return jnp.dot(inv_vand, jnp.dot(lagrange, ys))


def _paired_sums(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_0: Array,
    eq_1: Array,
    lam: Array,
) -> tuple[Array, Array, Array]:
    """Materialized `(s(0), 8*s(1/2), eq mass)` over the stride-2 pairs.

    s(0) reads the even elements at their eq weight; the u=1/2 sum works on
    doubled values (`e0 + e1 = 2*e(1/2)` per factor, likewise eq), which
    `_round_coeffs` rescales. Both go through the shared `LogupSummand` combine
    so the summand cannot drift from the verifier oracle's.
    """
    summand = LogupSummand(lam)
    scalars = summand.combine_scalars()
    eval_zero = jnp.sum(
        summand.combine(scalars, eq_0, n0[0::2], d1[0::2], n1[0::2], d0[0::2])
    )
    eq_h = eq_0 + eq_1
    eval_half = jnp.sum(
        summand.combine(
            scalars,
            eq_h,
            n0[0::2] + n0[1::2],
            d1[0::2] + d1[1::2],
            n1[0::2] + n1[1::2],
            d0[0::2] + d0[1::2],
        )
    )
    return eval_zero, eval_half, jnp.sum(eq_h)


def _expand_eq_slice(eval_point: Array, niv: int, *, row: bool) -> Array:
    """`expand_eq_to_hypercube` over the row (`eval_point[niv:]`) or batch
    (`eval_point[:niv]`) coordinate block, traced into the whole-layer jit. `niv`
    (and hence the slice bounds + output length) rides static."""
    coords = eval_point[niv:] if row else eval_point[:niv]
    return expand_eq_to_hypercube(coords, jnp.ones((), eval_point.dtype))


def pad_layer_to_capacity(
    layer: JaggedGkrLayer, capacities: tuple[int, ...]
) -> JaggedGkrLayer:
    """Re-store `layer` in a capacity layout: each segment extended to its
    capacity with the fold-neutral fraction (n=0, d=1), and `row_counts`
    becoming the capacity tuple.

    The prove over the capacity layer is byte-identical to the exact layout:
    the neutral fraction is a fixed point of the per-round fold (and of the
    re-pad gathers a non-even capacity's schedule inserts), and its eq mass
    moves from the closed-form virtual correction (`pad_adj - eq_sum`) into
    the materialized sum -- the round polynomials, challenges, and openings do
    not change. What changes is the compile-key surface: the whole-layer
    program's plane and schedule shapes now derive from `capacities` alone, so
    shards sharing a capacity tuple share every trace and executable, and the
    true row counts ride only in this one gather's runtime data.

    Any `capacities >= row_counts` works; the choice trades memory against
    cache hits. A memory-tight consumer keeps capacities at a running
    per-segment max over its shards (padding ~= the inter-shard spread); a
    power-of-two capacity additionally makes every fold even (no per-round
    re-pad gathers) at up to 2x padding."""
    if len(capacities) != len(layer.row_counts):
        raise ValueError(
            f"capacities {capacities} must have one entry per segment "
            f"({len(layer.row_counts)})"
        )
    for rc, cap in zip(layer.row_counts, capacities, strict=True):
        if cap < rc:
            raise ValueError(f"capacity {cap} < row count {rc}")
    gather = _segment_gather(layer.row_counts, capacities)
    n0, n1, d0, d1 = _pad_neutral(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        gather,
    )
    return JaggedGkrLayer(n0, n1, d0, d1, capacities)


def prove_jagged_layer(
    layer: JaggedGkrLayer,
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    *,
    challenge_limbs: int = 1,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """Run one jagged GKR layer's materialized sumcheck.

    `eval_point` is MSB-first over (batch || row) variables; its length
    fixes the virtual row depth `nrv = len(eval_point) - niv`, which may
    exceed what the materialized row counts need -- the extra rounds fold
    saturated all-ones segments against re-padded neutral rows, exactly the
    virtual positions' values. Returns the bound point (MSB-first, i.e. the
    challenges reversed), the advanced transcript, and the proof.
    """
    niv = layer.num_batch_variables
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    meta = _round_metadata(layer.row_counts, nrv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_from_meta(
        planes, niv, meta, lam, claim, eval_point, transcript, challenge_limbs
    )


def _check_row_space(row_counts: tuple[int, ...], num_vars: int, niv: int) -> int:
    """Validate the layer fits the virtual row space and return `nrv`. Host-side
    (Python-int) checks, kept out of the trace so the whole-layer jit never keys
    on `row_counts`."""
    nrv = num_vars - niv
    if nrv < 1:
        raise ValueError(
            f"eval_point must carry at least one row variable: got "
            f"{num_vars} coordinates for {niv} batch variables"
        )
    if max(row_counts) > 1 << nrv:
        raise ValueError(
            f"row count {max(row_counts)} exceeds the virtual row space "
            f"2^{nrv}; the row-eq lookup would run out of bounds"
        )
    return nrv


def _prove_jagged_layer_from_meta(
    planes: _Planes,
    niv: int,
    meta: list[tuple[Array | None, Array, Array]],
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    challenge_limbs: int,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """One jagged layer's sumcheck from a PREBUILT round schedule `meta`.

    `meta` is the per-round gather/index metadata, built host-side by
    `_round_metadata` and passed in as a traced operand rather than closed over the
    trace: its gathers span the layer height (hundreds of MB of int32 at shard
    scale), and baking them in as HLO constants is what made the whole-layer jit
    recompile from scratch per shard. As operands they leave the HLO tiny, so the
    compile is cheap and `row_counts` never enters the jit key."""
    nrv = eval_point.shape[0] - niv
    eq_row = _expand_eq_slice(eval_point, niv, row=True)
    eq_int = _expand_eq_slice(eval_point, niv, row=False)
    naturals, inv_vand = _round_interp_constants(eval_point.dtype)

    state = _JaggedState(planes, eq_row, eq_int, eval_point, lam, claim)
    sched = _JaggedSchedule(
        meta, _InterpConsts(naturals, inv_vand), nrv, niv, challenge_limbs
    )
    # The host round loop runs one fold-then-compute kernel per round, the FS hop
    # + reduce dispatching between them. `export_dispatch=True` selects the cached
    # per-round `jax.export` binary, but it only fires when this layer runs OUTSIDE
    # an outer jit (`JaggedGkrLayerRound(jit=False)`): the operands are then concrete
    # arrays, so each round host-dispatches and releases its buffers, bounding peak
    # host RAM on wide shards. Under the production outer jit
    # (`JaggedGkrLayerRound(jit=True)`) the dispatch sees tracers and falls back to
    # the eager kernel, tracing the whole loop into one program (the whole-scan
    # `zorch.sumcheck` megakernel was retired -- it never compiled at real sizes,
    # mirroring #332's drop of the dense megakernel).
    out = _run_jagged_rounds(state, sched, transcript, export_dispatch=True)
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


def _run_jagged_rounds_reference(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The unrolled oracle for `_run_jagged_rounds`: the per-round jagged sumcheck
    written out with an explicit observe/sample per round. Returns the bound point
    (challenges reversed), the advanced transcript, the stacked round polynomials,
    and the four folded pair openings. The round runner must match this byte-for-byte.
    """
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    naturals, inv_vand = sched.consts.naturals, sched.consts.inv_vand
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    point = eval_point
    polys: list[Array] = []
    challenges: list[Array] = []
    for rnd in range(nrv + niv):
        in_rows = rnd < nrv
        if in_rows:
            gather, col_index, pair_index = meta[rnd]
            n0, n1, d0, d1 = _pad_neutral(n0, n1, d0, d1, gather)
            w = eq_int[col_index]
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0,
                n1,
                d0,
                d1,
                eq_row[pair_index * 2] * w,
                eq_row[pair_index * 2 + 1] * w,
                lam,
            )
        else:
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0, n1, d0, d1, eq_int[0::2], eq_int[1::2], lam
            )
        poly = _round_coeffs(
            eval_zero,
            eval_half,
            eq_sum,
            eq_adj,
            pad_adj,
            point[-1],
            claim,
            naturals,
            inv_vand,
        )
        transcript = transcript.observe(poly)
        transcript, r = sample_challenge(transcript, claim.dtype, challenge_limbs)
        polys.append(poly)
        challenges.append(r)

        claim, pad_adj = _fold_scalars(poly, r, pad_adj, point[-1], one)
        n0, n1, d0, d1 = (_bind_lsb(a, r) for a in (n0, n1, d0, d1))
        if in_rows:
            eq_row = _bind_lsb(eq_row, r)
            if rnd == nrv - 1:
                # Rows exhausted: the accumulated row-eq product becomes the
                # scalar factor of every batch round; pad_adj restarts
                # to track the batch variables' own bound mass.
                eq_adj = pad_adj
                pad_adj = one
        else:
            eq_int = _bind_lsb(eq_int, r)
        point = point[:-1]

    return (
        jnp.stack(challenges[::-1]),
        transcript,
        jnp.stack(polys),
        n0[0],
        n1[0],
        d0[0],
        d1[0],
    )


# ===== per-round jagged sumcheck engine =====
# The per-round compute kernels + the host loop that threads them; each round's
# compute + device Fiat-Shamir hop traces into the whole-layer jit.


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["n0", "n1", "d0", "d1"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _Planes:
    """The four LogUp MLE planes (numerator_0/1, denominator_0/1) as one pytree --
    they travel and bind together through every round. A registered pytree so it
    crosses the whole-layer jit boundary as a single structured operand."""

    n0: Array
    n1: Array
    d0: Array
    d1: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["eq_adj", "pad_adj", "z_cur", "claim", "lam"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _RoundScalars:
    """The per-round scalar inputs to the round univariate: the eq/pad bound-mass
    corrections (`eq_adj`/`pad_adj`), the current point coordinate `z_cur`, the
    running `claim`, and the LogUp RLC coefficient `lam`. Scalar operands of the
    exported round kernel (a registered pytree)."""

    eq_adj: Array
    pad_adj: Array
    z_cur: Array
    claim: Array
    lam: Array


@dataclass(frozen=True)
class _InterpConsts:
    """The Lagrange interpolation constants (the `{0..DEGREE}` natural domain and
    the inverse Vandermonde). They depend only on dtype, so the round kernels bake
    them in as closure constants -- NOT export operands, so not a pytree."""

    naturals: Array
    inv_vand: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["planes", "eq_row", "eq_int", "eval_point", "lam", "claim"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _JaggedState:
    """A jagged layer's sumcheck carry: the four MLE planes, the row/batch
    eq tables, the bound point, the RLC `lam`, and the running `claim`. Bundled so
    the round-loop functions take one state instead of nine loose arrays -- the
    `(round, state, transcript)` shape `sumcheck.prove` and jagged-pcs's
    `_InnerState` already use."""

    planes: _Planes
    eq_row: Array
    eq_int: Array
    eval_point: Array
    lam: Array
    claim: Array


@dataclass(frozen=True)
class _JaggedSchedule:
    """A jagged layer's static round schedule: the per-round gather/index metadata,
    the interpolation constants, and the batch/row variable counts plus the
    challenge limb count. Rides beside the state so the loop signatures stay
    `(state, schedule, transcript)` rather than a 16-positional-arg bag."""

    meta: list[tuple[Array | None, Array, Array]]
    consts: _InterpConsts
    nrv: int
    niv: int
    challenge_limbs: int


def _bind_planes(planes: _Planes, alpha: Array) -> _Planes:
    return _Planes(
        *(_bind_lsb(a, alpha) for a in (planes.n0, planes.n1, planes.d0, planes.d1))
    )


def _round_poly_int(
    planes: _Planes, eq_int: Array, scalars: _RoundScalars, consts: _InterpConsts
) -> Array:
    """The `sum_as_poly` step for the dense batch phase: the round
    univariate from the current state, no fold (the entry kernel of a
    round loop, before any challenge is bound).

    Fiat-Shamir-less by construction — the FS hop lives in `_fs_reduce`, appended
    after the round compute — so the body is pure field arithmetic. `eq_int` is
    sliced stride-2 once inside `_paired_sums`."""
    eval_zero, eval_half, eq_sum = _paired_sums(
        planes.n0,
        planes.n1,
        planes.d0,
        planes.d1,
        eq_int[0::2],
        eq_int[1::2],
        scalars.lam,
    )
    return _round_coeffs(
        eval_zero,
        eval_half,
        eq_sum,
        scalars.eq_adj,
        scalars.pad_adj,
        scalars.z_cur,
        scalars.claim,
        consts.naturals,
        consts.inv_vand,
    )


def _fix_and_sum_int(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The `fix_and_sum` step for the dense batch phase: bind the previous
    round's challenge `alpha` (state size `m -> m/2`) **then** compute the next
    round's univariate at the halved size. Returns `(poly, planes, eq_int)` so the
    loop threads the folded state into the next round. The fold and the inner
    `_paired_sums` slice stride-2 twice."""
    planes = _bind_planes(planes, alpha)
    eq_int = _bind_lsb(eq_int, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _round_poly_row(
    planes: _Planes,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes]:
    """The row-variable round body shared by the row kernels: re-pad the four
    MLEs to the round's even layout (`gather`), look the per-pair batch eq
    weight up via `eq_int[col_index]`, and form the round univariate over the
    segment-local `eq_row` pairs. Returns `(poly, planes)` — the padded state the
    caller binds next round.

    The schedule (`gather`, `col_index`, `pair_index`) is host-built; the post-pad
    state is `gather`'s length (even), so the `_paired_sums` stride-2 stays valid."""
    n0, n1, d0, d1 = _pad_neutral(planes.n0, planes.n1, planes.d0, planes.d1, gather)
    w = eq_int[col_index]
    eval_zero, eval_half, eq_sum = _paired_sums(
        n0,
        n1,
        d0,
        d1,
        eq_row[pair_index * 2] * w,
        eq_row[pair_index * 2 + 1] * w,
        scalars.lam,
    )
    poly = _round_coeffs(
        eval_zero,
        eval_half,
        eq_sum,
        scalars.eq_adj,
        scalars.pad_adj,
        scalars.z_cur,
        scalars.claim,
        consts.naturals,
        consts.inv_vand,
    )
    return poly, _Planes(n0, n1, d0, d1)


def _fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The `fix_and_sum` step for the row-variable phase: bind the previous
    round's challenge `alpha` (state `2p_prev -> p_prev`, and `eq_row` halved in
    step) **then** re-pad to this round's layout and compute the next univariate.
    Returns `(poly, planes, eq_row)`.

    `eq_row` folds inside the kernel because the loop binds it at each round's
    end. The input state and `eq_row` enter even, and the `_pad_neutral` output is
    even, so all halvings stay valid."""
    planes = _bind_planes(planes, alpha)
    eq_row = _bind_lsb(eq_row, alpha)
    poly, planes = _round_poly_row(
        planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
    )
    return poly, planes, eq_row


def _fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The row->batch handoff in one launch: bind the last row variable's
    challenge `alpha` (the padded row state collapses to the dense batch
    state) **then** compute the first batch round's univariate over the
    still-unfolded `eq_int`.

    This is the one round whose fold is row-shaped (no `eq_int` bind) while its
    sum is batch-shaped. `eq_int` rides through unchanged; the batch
    rounds bind it from the next round on."""
    planes = _bind_planes(planes, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _fix_last(planes: _Planes, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """The `fix_last` step: bind the final challenge and read off the four pair
    openings (the fully-folded length-1 state's single element). `_finalize_layer`
    fuses this into the whole-layer kernel -- a fold step lowering to one kernel is
    the fusion-by-construction rule, not just a speedup."""
    p = _bind_planes(planes, alpha)
    return p.n0[0], p.n1[0], p.d0[0], p.d1[0]


@cache
def _round_interp_constants(dtype: Any) -> tuple[Array, Array]:
    """Lagrange `naturals` ({0..DEGREE}) and the inverse-Vandermonde, hoisted once
    per dtype. Both depend only on `_DEGREE`, so rebuilding them inside every
    `prove_jagged_layer` trace is pure redundant host work -- `compute_inv_vandermonde`
    is an O(DEGREE^2) numpy coefficient build, redone per GKR layer without the memo."""
    # Force concrete eval: `@cache` memoizes the result, so building it inside a
    # jit trace (the jit=True round zone) would cache a tracer that then escapes the
    # trace (UnexpectedTracerError). The constants are trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, dtype)
    return naturals, inv_vand


# Exported per-round kernels, keyed by the operand signature so one binary
# serves every round size in its bracket and is reused across rounds, layers, and
# shards (the recompile-free dispatch). Only the per-round-REPEATED variants are
# cached here -- `fix_and_sum_row` (the row rounds) and `fix_and_sum_int` (the
# interaction rounds); `sum_as_poly_row` / `_boundary` / `_last` fire once a layer
# and stay eager. The state dtype is part of the key: a multi-limb sumcheck folds
# base->extension after round 0, so the row binary is dispatched at two input
# dtypes (numerator base-field, denominator extension-field).
#
# The bare `Exported` is cached, not `jax.jit(exported.call)`: jit-wrapping cuts
# the per-call host dispatch (cached exec vs bare's per-call re-specialize) but is
# wall-clock NEUTRAL on the real prove -- the round dispatch overlaps async GPU/FS
# work, so the saved host time is off the critical path. Not worth the extra layer.
_ROUND_KERNEL_CACHE: dict[tuple, export.Exported] = {}

# Opt-in on-disk cache for the exported round binaries (set ZORCH_EXPORT_CACHE_DIR):
# their jax.export BUILD (symbolic StableHLO generation) re-runs every process and
# dominates the cold start, which the XLA persistent compile cache does NOT cover.
# Namespaced by jax version + a hash of every module the exported kernels close
# over (see `_export_cache_dir`), so any kernel-arithmetic edit invalidates it.
# Unset -> unchanged in-memory behaviour.
_EXPORT_CACHE_DIR = os.environ.get("ZORCH_EXPORT_CACHE_DIR")


@cache
def _export_cache_dir() -> Path:
    import hashlib

    import zorch.logup_gkr.circuit as _circuit
    import zorch.logup_gkr.prover as _prover
    import zorch.poly.eq as _eq
    import zorch.poly.univariate as _univariate

    # Hash every module whose code the exported round kernels close over, so any
    # edit to the round arithmetic invalidates the on-disk binary: this module, the
    # eq / circuit plane builders, `logup_combine` (the summand) in logup_gkr.prover,
    # and the Lagrange/Vandermonde interpolation in poly.univariate. Miss one and a
    # stale binary silently emits the OLD arithmetic — a wrong proof.
    h = hashlib.sha256()
    for mod in (_circuit, _eq, _prover, _univariate):
        src = mod.__file__
        assert src is not None  # imported modules always carry a source path
        h.update(Path(src).read_bytes())
    h.update(Path(__file__).read_bytes())
    d = (
        # Reached only under the `_EXPORT_CACHE_DIR is not None` guards in
        # `_round_get`/`_round_put`, so the env var is a real path string here.
        Path(_EXPORT_CACHE_DIR)  # type: ignore[arg-type]
        / f"{jax.__version__}-{h.hexdigest()[:12]}"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _export_path(key: tuple) -> Path:
    import hashlib

    return (
        _export_cache_dir()
        / f"{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.bin"
    )


def _round_get(key: tuple) -> export.Exported | None:
    exp = _ROUND_KERNEL_CACHE.get(key)
    if exp is None and _EXPORT_CACHE_DIR is not None:
        path = _export_path(key)
        if path.exists():
            exp = export.deserialize(bytearray(path.read_bytes()))
            _ROUND_KERNEL_CACHE[key] = exp
    return exp


def _round_put(key: tuple, exp: export.Exported) -> None:
    _ROUND_KERNEL_CACHE[key] = exp
    if _EXPORT_CACHE_DIR is not None:
        # Atomic publish: write a per-pid sibling temp then os.replace into place,
        # so a process sharing ZORCH_EXPORT_CACHE_DIR never deserializes a
        # half-written .bin (rename is atomic within one filesystem).
        path = _export_path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(bytes(exp.serialize()))
        os.replace(tmp, path)


def _round_dispatch(
    key: tuple, operands: tuple, build: Callable[[], export.Exported]
) -> Any:
    """The shared round export-cache dispatch every `_dispatch_*` runs: reuse the
    cached binary for `key`, else `build()` it (the symbolic export is the cold
    cost, so only on a miss) and cache it, then call it on the concrete `operands`.
    The tracer fallback, `operands`, `key`, and the abstract shapes stay per-round;
    only this get / export / put / call protocol is shared.

    The call binds `call_exported_p` directly rather than going through
    `Exported.call`: that method wraps every invocation in a `custom_vjp` (its
    `f_imported`/`f_flat` AD path), which the eager host round loop never
    differentiates. Skipping it is ~177us -> ~55us warm per dispatch (a `jax.jit`
    dispatch costs the same) on the dispatch-bound host-FS prove, with the SAME one
    symbolic binary -- no per-shape recompile, byte-identical (same primitive, same
    flat operands)."""
    exported = _round_get(key)
    if exported is None:
        exported = build()
        _round_put(key, exported)
    flat = jax.tree_util.tree_leaves(operands)
    return exported.out_tree.unflatten(_call_exported_p.bind(*flat, exported=exported))


# `_Planes` / `_RoundScalars` cross the jax.export boundary as pytree operands;
# register their (empty -- no meta_fields) aux so `Exported.serialize()` can
# round-trip them for the on-disk cache above.
if _EXPORT_CACHE_DIR is not None:
    for _t in (_Planes, _RoundScalars):
        try:
            export.register_pytree_node_serialization(
                _t,
                serialized_name=f"zorch.logup_gkr.jagged_prover.{_t.__name__}",
                serialize_auxdata=lambda _a: b"",
                deserialize_auxdata=lambda _b: (),
            )
        except ValueError:
            # Idempotent across a re-import (importlib.reload): serialized_name is
            # a constant, so jax raises "Duplicate serialization registration".
            # The prior, identical registration is already live -- keep it.
            pass

# The symbolic bound only needs to *contain* every round size (`exported.call`
# re-specializes XLA codegen per concrete size regardless), but it MUST exceed the
# largest dispatched state: a row input is `2*pp` and an interaction input `4*g`,
# so the bound caps the provable layer at `2*_ROUND_SYM_MAX` / `4*_ROUND_SYM_MAX`
# elements. Hold it well above any trace's 2^(log rows) so no real shard overflows.
_ROUND_SYM_MAX = 1 << 30


_SCALAR_FIELDS = ("eq_adj", "pad_adj", "z_cur", "claim", "lam")


def _abst_scalars(scalars: _RoundScalars) -> _RoundScalars:
    return _RoundScalars(
        *(jax.ShapeDtypeStruct((), getattr(scalars, f).dtype) for f in _SCALAR_FIELDS)
    )


def _dispatch_fix_and_sum_int(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the dense interaction round through one cached binary symbolic
    over the halving state size. The round folds (`m -> m/2`) then `_paired_sums`
    slices stride-2 again (`m/2 -> m/4`), two halvings with no re-pad between, so
    the state is `4*g` to keep both decidable; `eq_int` halves with it.

    `exported.call` is a host dispatch; under a `jax.jit` trace the operands are
    tracers, so fall back to the eager kernel -- the jit compiles the round
    itself, the per-round export being its alternative."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_int(planes, eq_int, alpha, scalars, consts)
    operands = (planes, eq_int, alpha, scalars)
    # Per-operand dtypes (a LogUp numerator is base-field, its denominator
    # extension-field, and the state promotes base->extension across rounds), so
    # each (round-shape, dtype-mix) gets its own binary; `consts` is baked in.
    key = (
        "int",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )

    def build() -> export.Exported:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((4 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda p, e, al, sc: _fix_and_sum_int(p, e, al, sc, consts)  # noqa: E731
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the row->interaction handoff (bind the last row challenge `alpha`,
    then sum the first interaction round over the still-unfolded `eq_int`) through
    one cached binary. Mirrors `_dispatch_fix_and_sum_int` without the `eq_int`
    bind: the bind halves the state (`4*g -> 2*g`) and `eq_int` rides unfolded at
    `2*g` (= the post-bind state), so one dispatched kernel replaces the eager one.
    """
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_boundary(planes, eq_int, alpha, scalars, consts)
    operands = (planes, eq_int, alpha, scalars)
    key = (
        "boundary",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )

    def build() -> export.Exported:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda p, e, al, sc: _fix_and_sum_boundary(
            p, e, al, sc, consts
        )  # noqa: E731
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_sum_as_poly_row(
    planes: _Planes,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes]:
    """Dispatch the round-0 sum (no fold, no challenge) through one cached binary
    symbolic over the RAW layer height (`h`), the re-pad layout (`2*p` gather / `p`
    schedule), and `eq_row` (`2*rr`); `eq_int` rides fixed. Mirrors
    `_dispatch_fix_and_sum_row` without the bind -- one dispatched kernel replaces the
    eager entry kernel. A round needing no re-pad (`gather` None) gets an identity
    gather over the full height (no `//2` fold) so it hits the same binary."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _round_poly_row(
            planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
        )
    if gather is None:
        gather = jnp.arange(planes.n0.shape[0], dtype=col_index.dtype)
    operands = (planes, gather, col_index, pair_index, eq_row, eq_int, scalars)
    key = (
        "sum0",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        eq_int.shape,
        consts.naturals.shape[0],
    )

    def build() -> export.Exported:
        h, p, rr = export.symbolic_shape(
            "h, p, rr",
            constraints=[
                "h >= 1",
                f"h <= {_ROUND_SYM_MAX}",
                "p >= 1",
                f"p <= {_ROUND_SYM_MAX}",
                "rr >= 1",
                f"rr <= {_ROUND_SYM_MAX}",
            ],
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((h,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * p,), gather.dtype),
            jax.ShapeDtypeStruct((p,), col_index.dtype),
            jax.ShapeDtypeStruct((p,), pair_index.dtype),
            jax.ShapeDtypeStruct((2 * rr,), eq_row.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda pl, ga, ci, pi, er, ei, sc: _round_poly_row(  # noqa: E731
            pl, ga, ci, pi, er, ei, sc, consts
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """Dispatch a jagged row round through one cached binary symbolic over the
    input state (`2*pp`), the re-pad layout (`2*p` gather / `p` schedule), and the
    halving `eq_row` (`2*rr`); `eq_int` rides at its fixed `2^niv` width. A round
    that needs no re-pad (gather `None`) gets an identity gather so it hits the
    same binary -- the identity pad is a no-op, so byte-identical."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_row(
            planes,
            eq_row,
            alpha,
            gather,
            col_index,
            pair_index,
            eq_int,
            scalars,
            consts,
        )
    if gather is None:
        gather = jnp.arange(planes.n0.shape[0] // 2, dtype=col_index.dtype)
    operands = (planes, eq_row, alpha, gather, col_index, pair_index, eq_int, scalars)
    key = (
        "row",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        eq_int.shape,
        consts.naturals.shape[0],
    )

    def build() -> export.Exported:
        pp, p, rr = export.symbolic_shape(
            "pp, p, rr",
            constraints=[
                "pp >= 1",
                f"pp <= {_ROUND_SYM_MAX}",
                "p >= 1",
                f"p <= {_ROUND_SYM_MAX}",
                "rr >= 1",
                f"rr <= {_ROUND_SYM_MAX}",
            ],
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((2 * pp,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * rr,), eq_row.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            jax.ShapeDtypeStruct((2 * p,), gather.dtype),
            jax.ShapeDtypeStruct((p,), col_index.dtype),
            jax.ShapeDtypeStruct((p,), pair_index.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda pl, er, al, ga, ci, pi, ei, sc: _fix_and_sum_row(  # noqa: E731
            pl, er, al, ga, ci, pi, ei, sc, consts
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _fold_scalars(
    poly: Array, r: Array, pad_adj: Array, z: Array, one: Array
) -> tuple[Array, Array]:
    """The per-round scalar fold: the next claim (round poly evaluated at `r`) and the
    updated pad-mass `pad_adj`. One source for both the oracle
    `_run_jagged_rounds_reference` (which inlines it) and the round loop's
    `_reduce_body`, so the two cannot drift out of byte-equality."""
    return eval_coeffs(poly, r), pad_adj * (z * r + (one - z) * (one - r))


def _reduce_body(
    raw: Array,
    poly: Array,
    pad_adj: Array,
    z_cur: Array,
    one: Array,
    eval_point: Array,
    pos: Array,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, Array]:
    """Reinterpret the squeezed challenge, fold the round scalars, AND slice the
    next round's eval-point coordinate. Three hops collapse here: the challenge
    reshape/bitcast, the scalar fold, and the per-round `eval_point` gather (a
    `jnp.take` is a real ~22us dispatch, NOT a buffer view). `pos` indexes this
    round's coordinate; the next is `pos - 1`, threaded device-resident so no
    per-round index round-trips the host. Returns the round challenge `r`, the next
    `claim`, `pad_adj`, the next round's `z_cur`, and the decremented `pos`. Plain
    (un-jitted) so it fuses into whichever kernel owns it -- the round loop's
    `_fs_reduce`, which prepends the Fiat-Shamir hop."""
    r = reinterpret_challenge(raw, dtype)
    claim, pad_adj = _fold_scalars(poly, r, pad_adj, z_cur, one)
    # The last round's `pos_next` is -1 (a dead output -- no round consumes it);
    # clamp so the slice index is provably in-bounds rather than leaning on
    # `dynamic_slice`'s implicit index clamp. No-op for every live round (pos >= 1).
    pos_next = jnp.maximum(pos - 1, jnp.int32(0))
    z_next = jax.lax.dynamic_index_in_dim(eval_point, pos_next, keepdims=False)
    return r, claim, pad_adj, z_next, pos_next


def _fs_reduce(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    eval_point: Array,
    pos: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array, Array, Array]:
    """The per-round FS hop + reduce: observe `poly`, squeeze the challenge, then
    `_reduce_body`. Returns the advanced transcript and `(r, claim, pad_adj, z_next,
    pos_next)`. No jit of its own -- it fuses into the round's compute under the
    whole-layer jit. `one` is baked.

    The device FS hop rides the `zorch.duplex_fs` composite
    (`observe_and_sample_marked`) so the whole absorb+squeeze lowers to ONE
    register-resident kernel. Without the marker the duplex glue (rate-block merge,
    position select, output extract) decomposes into ~6k loop-fused ops/hop,
    dominating the layer compile; the generic fused_region path is declined by the
    vendor (exponential LoopFusion), so the dedicated `zorch.duplex_fs` emitter is
    what fuses it."""
    transcript, raw = observe_and_sample_marked(transcript, poly, n)
    one = jnp.ones((), dtype)
    r, claim, pad_adj, z_next, pos_next = _reduce_body(
        raw, poly, pad_adj, z_cur, one, eval_point, pos, dtype
    )
    return transcript, r, claim, pad_adj, z_next, pos_next


# The layer tail: the final fold (`_fix_last`) plus stacking the per-round
# challenge/poly lists. Folding `_fix_last` in here keeps the final fold in the
# whole-layer kernel without decorating the bare helper.
def _finalize_layer(
    planes: _Planes, alpha: Array, chal: list[Array], poly: list[Array]
) -> tuple[Array, Array, Array, Array, Array, Array]:
    fn0, fn1, fd0, fd1 = _fix_last(planes, alpha)
    return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
    *,
    export_dispatch: bool = False,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-layer device-FS sumcheck host loop: one fold-then-compute per round
    at the round's real (halving) state size, the Fiat-Shamir hop + reduce folded in
    per round. One `sum_as_poly` (round 0, no fold), one `fix_and_sum` per subsequent
    round (row / boundary / batch variant by round index), one `fix_last`.

    On the default path this runs under the consumer's whole-layer `jax.jit`: every
    round's compute + FS hop traces into one fused layer kernel (the per-round host
    dispatches collapse to one per layer).

    `export_dispatch` selects, per round, the cached `jax.export` binary
    (`_dispatch_*`, one symbolic-size kernel host-relaunched at the halving size)
    over the eager kernel. It only fires when this loop runs OUTSIDE an outer
    `jax.jit` -- the operands are then concrete arrays, not tracers, so
    `exported.call` host-dispatches each round (the FS hop + reduce dispatching
    eagerly between rounds) and releases its buffers before the next, bounding
    peak host RAM (the decoupled production path). Under the outer jit the
    dispatch sees tracers and falls back to the eager kernel, tracing the whole
    loop into one program. Both paths are byte-identical to the inline reference
    oracle in the tests (same math; the export path only regroups it across
    per-round host dispatches)."""
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    planes = state.planes
    consts = sched.consts
    transcript = cast(DuplexTranscript, transcript)

    # The dispatch and eager kernels share signatures, so select one per round.
    fix_row = _dispatch_fix_and_sum_row if export_dispatch else _fix_and_sum_row
    fix_int = _dispatch_fix_and_sum_int if export_dispatch else _fix_and_sum_int
    fix_boundary = (
        _dispatch_fix_and_sum_boundary if export_dispatch else _fix_and_sum_boundary
    )
    # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
    sum0 = _dispatch_sum_as_poly_row if export_dispatch else _round_poly_row
    polys: list[Array] = []
    challenges: list[Array] = []
    prev_r = one  # unused until the first fold (round 1)
    # z_cur is eval_point's coordinate for round k (== eval_point[-(k+1)]). Rather
    # than a standalone `jnp.take` every round (a real ~22us gather dispatch, not a
    # free buffer view), the coordinate is threaded device-resident: round 0 reads
    # the last coordinate and each `_reduce_body` slices the next via a
    # decremented `pos`, riding the fold's dispatch instead of its own. The fold stays
    # on the compute device (a host CPU reduce forces the carry to round-trip back to
    # GPU before each bind, which serializes the bind pipeline -- net slower).
    pos = jnp.asarray(eval_point.shape[0] - 1, jnp.int32)
    z_cur = jnp.take(eval_point, -1)
    for rnd in range(nrv + niv):
        scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
        dtype = claim.dtype
        if rnd == 0:
            gather, col_index, pair_index = meta[0]
            poly, planes = sum0(
                planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
            )
        elif rnd < nrv:
            gather, col_index, pair_index = meta[rnd]
            poly, planes, eq_row = fix_row(
                planes,
                eq_row,
                prev_r,
                gather,
                col_index,
                pair_index,
                eq_int,
                scalars,
                consts,
            )
        elif rnd == nrv:
            poly, planes, eq_int = fix_boundary(planes, eq_int, prev_r, scalars, consts)
        else:
            poly, planes, eq_int = fix_int(planes, eq_int, prev_r, scalars, consts)
        # Device FS hop + reduce -- traced into the whole-layer jit on the default
        # path (one fused region per round), dispatched eagerly between rounds on
        # the export path. Slices the next z_cur via the decremented `pos`, riding
        # the fold's dispatch instead of a standalone gather.
        transcript, r, claim, pad_adj, z_cur, pos = _fs_reduce(
            poly, transcript, pad_adj, z_cur, eval_point, pos, challenge_limbs, dtype
        )
        polys.append(poly)
        challenges.append(r)
        if rnd == nrv - 1:
            eq_adj = pad_adj
            pad_adj = one
        prev_r = r

    fn0, fn1, fd0, fd1, stacked_challenges, stacked_polys = _finalize_layer(
        planes, prev_r, challenges, polys
    )
    return (
        stacked_challenges,
        transcript,
        stacked_polys,
        fn0,
        fn1,
        fd0,
        fd1,
    )


# The inter-layer carry: sample `lam` + the batched claim before the round loop,
# absorb the openings + sample + fold the child selector after. Pure device math
# bracketing the layer's FS samples; traces into the whole-layer round zone.
def _sample_lam_and_claim(
    transcript: DuplexTranscript,
    num_eval: Array,
    den_eval: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array]:
    """The layer pre-carry: squeeze the batching `lam`, reinterpret it, and form the
    claim `lam*num_eval + den_eval`. All device math, traced into the whole-layer
    zone."""
    transcript, raw = transcript.sample(n)
    lam = reinterpret_challenge(raw, dtype)
    return transcript, lam, lam * num_eval + den_eval


def _observe_openings_and_fold(
    transcript: DuplexTranscript,
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    point: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array]:
    """Device-FS layer post-carry: absorb the four openings, squeeze the child
    selector `r`, and fold the carry. The openings stack, `observe`, `sample`,
    reinterpret, and `fold_carry` are all device math that trace into the whole-layer
    zone -- the layer-boundary sibling of the per-round FS hop. `observe_and_sample`
    fuses the absorb + squeeze exactly as the round FS does, so the transcript stream
    is byte-identical to the split form."""
    transcript, raw = transcript.observe_and_sample(jnp.stack([n0, n1, d0, d1]), n)
    r = reinterpret_challenge(raw, dtype)
    return transcript, *fold_carry(n0, n1, d0, d1, point, r)


def _prove_jagged_layer_round(
    planes: _Planes,
    niv: int,
    meta: list[tuple[Array | None, Array, Array]],
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """One jagged GKR layer's carry reduction: sample the batching `lam`, prove
    the layer, observe the openings, and fold the carry with the child selector.

    Takes the planes + batch count + prebuilt `meta` (not a `JaggedGkrLayer`)
    so the whole-layer jit never keys on `row_counts` and never bakes the schedule
    into the trace. A module-level function (no implicit `self`) so the chain can
    drop a round -- and free its layer -- the moment it builds the next (the
    one-live-layer release `ChainedJaggedProveTest` pins)."""
    num_eval, den_eval, eval_point = carry
    dtype = num_eval.dtype
    transcript = cast(DuplexTranscript, transcript)
    # The per-layer carry brackets the round loop: sample lam + the batched claim
    # before, absorb the openings + sample + fold the child selector after. All
    # device math, traced into the whole-layer jit.
    transcript, lam, claim = _sample_lam_and_claim(
        transcript, num_eval, den_eval, challenge_limbs, dtype
    )
    point, transcript, proof = _prove_jagged_layer_from_meta(
        planes, niv, meta, lam, claim, eval_point, transcript, challenge_limbs
    )
    n0, n1 = proof.numerator_0, proof.numerator_1
    d0, d1 = proof.denominator_0, proof.denominator_1
    transcript, num_eval, den_eval, eval_point = _observe_openings_and_fold(
        cast(DuplexTranscript, transcript),
        n0,
        n1,
        d0,
        d1,
        point,
        challenge_limbs,
        dtype,
    )
    return (num_eval, den_eval, eval_point), transcript, proof


# Shared by every `JaggedGkrLayerRound`. The schedule (`meta`) rides as a TRACED
# operand, not a static arg: its per-round gather/index arrays span the layer
# height (hundreds of MB of int32 at shard scale), so closing them over the trace
# baked them into the HLO as constants -- XLA then spent minutes folding those
# constants, a per-shard from-scratch compile that scaled with the trace. As
# operands the HLO carries no data, so the compile is small and height-independent
# (~15s whether the layer is 2M rows or 77M) and `row_counts` values leave the jit
# key: it keys only on the per-round operand SHAPES plus the static `niv` /
# `challenge_limbs` (`nrv` is read from `eval_point`'s length inside). Two layers
# still recompile when their shape sequence differs, but each compile is cheap and
# persistent-cached. Routing through one module-level zone lets freshly built
# same-shape rounds reuse a single trace, so a consumer rebuilding the chain each
# warm iteration (the generator keeping lazy one-live-layer release) re-traces at
# most per distinct shape sequence, not per iter.
@partial(jax.jit, static_argnums=(5, 6))
def _jagged_round_zone(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    meta: list[tuple[Array | None, Array, Array]],
    niv: int,
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    planes = _Planes(numerator_0, numerator_1, denominator_0, denominator_1)
    return _prove_jagged_layer_round(
        planes, niv, meta, challenge_limbs, carry, transcript
    )


def _jagged_round_via_zone(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Build the round schedule host-side and dispatch through `_jagged_round_zone`
    with the planes + `meta` as traced operands. Splitting `meta` out of the trace
    (rather than the layer's static `row_counts`) is what keeps the whole-layer
    compile shard-independent."""
    niv = layer.num_batch_variables
    eval_point = carry[2]
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    meta = _round_metadata(layer.row_counts, nrv)
    return _jagged_round_zone(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        meta,
        niv,
        challenge_limbs,
        carry,
        transcript,
    )


class JaggedGkrLayerRound(Round):
    """Prove one jagged GKR layer; the chain of these (floor outward) is the
    jagged GKR prover, threading the same `(num_eval, den_eval, eval_point)`
    carry as the dense chain. `challenge_limbs` rides on the round because
    every challenge in the layer -- lam, the per-variable folds, and the
    child-selector r -- must come from the same squeeze rule.

    The shared head `prover.bind_output` works unchanged for a jagged output
    when `challenge_limbs == 1`; a consumer squeezing multi-limb challenges
    owns its binding glue.

    The per-layer prove dispatches through the module-level `_jagged_round_zone`
    with the round schedule as a traced operand, so it keys on `(niv, plane
    shapes)` and never on `row_counts` -- the trace (and its compiled kernel) is
    reused across every same-shape round, and shards differing only in row counts
    share one compile. The round holds only its layer (no per-instance jit, no
    self-closure), so the chain's release bound is untouched. The pyramid stays a
    host-orchestrated Python loop of these (one trace per layer shape, never one
    `jit` over the whole pyramid -- it does not fit at scale; see
    `prover.LogupSumcheckRound`).
    """

    def __init__(self, layer: JaggedGkrLayer, challenge_limbs: int = 1) -> None:
        # `partial` closes over (layer, challenge_limbs), not `self`, so the chain
        # frees the round -- and its layer -- the moment it builds the next.
        self._call = partial(_jagged_round_via_zone, layer, challenge_limbs)

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
