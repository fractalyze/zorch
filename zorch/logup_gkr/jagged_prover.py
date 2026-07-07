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
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, export
from jax._src.export._export import call_exported_p as _call_exported_p

from zorch._composite import composite
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
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    SUMCHECK_ROUND_MARKER_VERSION,
    fold_pair,
)
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    Transcript,
    _state_leaves,
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


@dataclass(frozen=True)
class RoundWidthCaps:
    """Fixed round-buffer widths for the size-invariant jagged sumcheck
    (xla#179): with caps set, every round of a phase runs at one static
    operand shape -- the live prefix tracked by the round's `live` operand --
    so one compiled round kernel serves every round, layer, and shard under
    the caps. Hashable (a jit static arg on the per-layer round zone).

    `row` bounds the row-phase plane/gather width (>= the round-0 even-padded
    layout, a multiple of 4); `eq_row` bounds the row-eq table (>= 2^nrv,
    even); `interaction` bounds the dense-phase state and eq width (>= 2^niv,
    a multiple of 4)."""

    row: int
    eq_row: int
    interaction: int


def _round_metadata_impl(
    row_counts: tuple[int, ...], num_row_vars: int, width: int | None = None
) -> list[tuple[Array | None, Array, Array, Array]]:
    """Per-round `(gather, col_index, pair_index, live)` for the row phase.

    Memoized on the (static) layout for the EXACT (`width=None`) layout only:
    the schedule is a pure function of the Python-int row counts, so a cold
    trace reuses the device-resident index arrays across same-shape layers
    instead of rebuilding them (host numpy + a batched device_put); those
    arrays are tiny, so caching them costs negligible device memory. Capped
    (`width` set) layouts are NOT memoized: their keys are the per-layer exact
    row counts (no reuse within a shard's halving chain), and every index
    array is laid out at the full `width` -- a never-evicting cache of those
    is a device-memory leak that grows with the layer count.

    Round k folds the layout round k-1 left behind: odd segments pre-pad to
    even (`gather`; None when already even), then the stride-2 fold halves
    every segment. `col_index` maps each pair to its batch element and
    `pair_index` to its in-segment pair offset -- the eq_row lookup is
    segment-local because a jagged layer is batch-major, so the row-eq
    factor is indexed per segment while `eq_int[col_index]` carries the
    batch weight. All static, derived from the Python-int row counts.

    `live` is the round's i32[2] `{live pairs, live eq_row}` operand -- on the
    exact layout it just restates the schedule widths; with `width` set
    (`RoundWidthCaps.row`) the schedule is laid into fixed-width buffers
    (sentinel-padded gather via `_fixed_width_gather`, zero-padded indices)
    and `live` marks the prefix that is real.
    """
    # Build the whole schedule on the host, then commit it in ONE batched
    # device_put (not per round): every index array is tiny and static. The None
    # gathers (no re-pad) ride through device_put as empty pytree nodes.
    host_meta: list[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]] = []
    counts = row_counts
    for rnd in range(num_row_vars):
        padded, pairs = _prepad_folded(
            counts
        )  # the circuit's own prepad/fold recurrence
        col_index = np.repeat(np.arange(len(pairs), dtype=np.int32), pairs)
        pair_index = np.concatenate([np.arange(pc, dtype=np.int32) for pc in pairs])
        # live[1] = eq_row's live length ENTERING round `rnd`. Only the mid
        # rounds (1..) fold eq_row -- round 0 (first) reads it un-bound -- so
        # round k enters with 2^nrv halved k-1 times, not k. Only the claimed
        # kernel consumes this bound (it sizes the folded-eq_row write); the
        # decomposition folds the full width, so a wrong value here is
        # invisible on the decompose-inline (CPU) route and surfaces as
        # garbage eq_row reads a round later on the claimed one.
        live = np.asarray(
            [sum(pairs), (1 << num_row_vars) >> max(rnd - 1, 0)], dtype=np.int32
        )
        if width is None:
            gather = _segment_gather_np(counts, padded)
        else:
            if width % 2 or width < sum(padded):
                raise ValueError(
                    f"round width cap {width} cannot hold the round-{rnd} "
                    f"even-padded layout ({sum(padded)} slots; the cap must "
                    "be even)"
                )
            gather = _fixed_width_gather(counts, padded, width)
            col_index = _zero_pad_index_np(col_index, width // 2)
            pair_index = _zero_pad_index_np(pair_index, width // 2)
        host_meta.append((gather, col_index, pair_index, live))
        counts = pairs
    # `ensure_compile_time_eval` forces the device_put to materialize a CONCRETE
    # committed array: this memoized builder is hit inside the round-zone trace, and
    # without it the cached value is a trace-scoped `device_put` tracer that escapes
    # when a later call reuses the cache (UnexpectedTracerError). Concrete -> the jit
    # bakes it as a constant.
    with jax.ensure_compile_time_eval():
        return jax.device_put(host_meta)


_round_metadata_cached = cache(_round_metadata_impl)


def _round_metadata(
    row_counts: tuple[int, ...], num_row_vars: int, width: int | None = None
) -> list[tuple[Array | None, Array, Array, Array]]:
    """The per-round row schedule: memoized for the exact layout, built fresh
    for capped layouts (`_round_metadata_impl` has the full story).

    Marker v2 (xla#179 device-derived schedule) no longer carries these
    arrays — the claimed kernels and the composite decompositions derive the
    schedule from `row_counts` + the round index (`_derive_row_schedule`).
    This host builder remains the independent oracle's source
    (`_run_jagged_rounds_reference`) so the derivation stays cross-checked
    against the original construction."""
    if width is None:
        return _round_metadata_cached(row_counts, num_row_vars)
    return _round_metadata_impl(row_counts, num_row_vars, width)


@cache
def _round_live_meta(row_counts: tuple[int, ...], num_row_vars: int) -> list[Array]:
    """Per-round i32[3] `{live pairs, live eq_row entry, round}` operands for
    the v2 round markers — the only per-round schedule state that still rides
    the marker (the index arrays derive in place from `row_counts`). Values
    restate `_round_metadata_impl`'s recurrence; independent of any width cap
    (a cap changes buffer layout, never liveness), so one memoized list serves
    the exact and every capped layout. Tiny (3 ints/round), so the
    never-evicting cache is safe."""
    host = []
    counts = row_counts
    for rnd in range(num_row_vars):
        _, pairs = _prepad_folded(counts)
        # live[1] = eq_row liveness ENTERING round `rnd`: round 0 never folds
        # eq_row, so round k enters with 2^nrv halved k-1 times (see
        # _round_metadata_impl's note — only the claimed kernel reads this).
        host.append(
            np.asarray(
                [sum(pairs), (1 << num_row_vars) >> max(rnd - 1, 0), rnd],
                np.int32,
            )
        )
        counts = pairs
    with jax.ensure_compile_time_eval():
        return jax.device_put(host)


def _round_out_pairs(row_counts: tuple[int, ...], num_row_vars: int) -> tuple[int, ...]:
    """Per-round padded pair count `sum(pairs_k)` — the STATIC output width of
    an exact-layout round (`2 * out_pairs` padded slots). The capped layout is
    width-preserving instead (out width = the plane buffer width), so it never
    consumes this."""
    out = []
    counts = row_counts
    for _ in range(num_row_vars):
        _, pairs = _prepad_folded(counts)
        out.append(sum(pairs))
        counts = pairs
    return tuple(out)


def _derive_row_schedule(
    row_counts: Array, rnd: Array, num_pairs: int, sentinel: int, idx_dtype: Any
) -> tuple[Array, Array, Array]:
    """Round `rnd`'s `(gather, col_index, pair_index)` derived in-trace from
    the layer's `row_counts` — the traced mirror of `_round_metadata_impl`'s
    host build, and the byte-exact fallback contract for the v2 markers (the
    claimed kernels run the same derivation in place).

    Entering round k every segment holds `counts_k[s] = ceil(rc[s] / 2^k)`
    elements (iterated ceil-halving composes into one shift) and folds
    `pairs_k[s] = ceil(counts_k / 2)` pairs. A pair (segment s, in-segment
    pair pj) re-pads from source elements `cum_counts_k[s] + 2·pj / +1`; only
    the odd element of the last pair of an odd-count segment is the neutral
    pad. `num_pairs` is the STATIC output pair count (the fixed buffer's
    `width // 2`, or the exact layout's `sum(pairs_k)`); `sentinel` any index
    the pad blend treats as past-the-end (the folded state length). Dead
    slots past the live pairs carry the sentinel / zero, matching
    `_fixed_width_gather` / `_zero_pad_index_np`."""
    i32 = jnp.int32
    rc = row_counts.astype(i32)
    # ((rc − 1) >> k) + 1 = ceil(rc / 2^k) for rc >= 1, and 0 for an empty
    # segment under the arithmetic shift ((−1 >> k) + 1) — no pairs, so the
    # searchsorted below never lands on it.
    counts = (rc - i32(1) >> rnd.astype(i32)) + i32(1)
    pairs = counts + i32(1) >> 1
    cum_pairs = jnp.cumsum(pairs)  # inclusive; cum_pairs[-1] = live pairs
    seg_base = jnp.concatenate([jnp.zeros((1,), i32), jnp.cumsum(counts)[:-1]])
    pr = jnp.arange(num_pairs, dtype=i32)
    s = jnp.searchsorted(cum_pairs, pr, side="right").astype(i32)
    s = jnp.minimum(s, i32(row_counts.shape[0] - 1))  # dead-tail clamp
    pj = pr - (cum_pairs[s] - pairs[s])
    live = pr < cum_pairs[-1]
    j_e = pj * 2
    j_o = j_e + 1
    src_e = seg_base[s] + j_e
    src_o = seg_base[s] + j_o
    sent = i32(sentinel)
    gather_e = jnp.where(live, src_e, sent)
    gather_o = jnp.where(live & (j_o < counts[s]), src_o, sent)
    gather = jnp.stack([gather_e, gather_o], axis=1).reshape(-1)
    zero = i32(0)
    col_index = jnp.where(live, s, zero)
    pair_index = jnp.where(live, pj, zero)
    return (
        gather.astype(idx_dtype),
        col_index.astype(idx_dtype),
        pair_index.astype(idx_dtype),
    )


def _pad_to_width(arr: Array, width: int, neutral: int) -> Array:
    """Extend `arr` to `width` with the fold-neutral fraction tail -- 0 for a
    numerator, 1 for a denominator -- keeping the live prefix at the front. The
    fixed-width round-buffer convention (xla#179), so every round of a phase
    runs at one static shape."""
    pad = width - arr.shape[0]
    if pad == 0:
        return arr
    tail = jnp.zeros((pad,), arr.dtype) if neutral == 0 else jnp.ones((pad,), arr.dtype)
    return jnp.concatenate([arr, tail])


def _fixed_width_gather(
    src_counts: tuple[int, ...], dst_counts: tuple[int, ...], width: int
) -> np.ndarray:
    """`_segment_gather` laid into a fixed `width` buffer. `_segment_gather`'s
    intra-segment sentinel is `sum(src_counts)`: past the live rows in the
    exactly-sized layout, but a live slot in the wider buffer, so remap it (and
    any index past the live rows) to `width`, which the neutral-pad gather
    resolves to the neutral pad rather than a stale slot. `None` (layouts
    already agree) becomes the identity over the live prefix."""
    live = sum(src_counts)
    seg = _segment_gather_np(src_counts, dst_counts)
    base = np.arange(live, dtype=np.int32) if seg is None else seg
    base = np.where(base >= live, width, base)
    row = np.full(width, width, dtype=np.int32)
    row[: base.shape[0]] = base
    return row


def _zero_pad_index_np(arr: np.ndarray, width: int) -> np.ndarray:
    """Lay a host index array into a fixed-width buffer, zero tail. Zero keeps
    every pad slot in-bounds for its lookup; the reduce masks the tail dead by
    the round's `live` operand."""
    out = np.zeros(width, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def _resize_zero(arr: Array, width: int) -> Array:
    """Resize a fixed-width round buffer: slice the prefix down or zero-pad up.
    Only correct when the live prefix fits in `width` -- the tail past it is
    dead (masked by the rounds' `live` operand)."""
    if arr.shape[0] >= width:
        return arr[:width]
    return _pad_to_width(arr, width, 0)


def _bind_lsb(arr: Array, r: Array) -> Array:
    """Bind the LSB variable: stride-2 consecutive pairs fold via the shared
    `sumcheck.prover.fold_pair` -- `e0 + r*(e1 - e0)`. (The split is LSB/stride-2,
    distinct from `fold`/`split_halves`' contiguous MSB halves; only the scalar
    fold is shared.)"""
    return fold_pair(arr[0::2], arr[1::2], r)


# Layer-entry donated cap buffers (xla#179 pad donation). Under a machine
# cap the entry pad materializes FRESH cap-wide buffers every layer -- an
# alloc + zero-fill + prefix copy per plane/eq table, ~2x cap-width writes,
# the top GPU item of the warm decoupled prove (wrapped_concatenate +
# wrapped_broadcast at a cap that is ~19x shard17's live prefix). The pool
# instead holds ONE persistent cap-wide array per (role, width, dtype); each
# layer donates it back to `_lay_prefix`, which writes only the live prefix
# in place. The tail keeps the PREVIOUS layer's bytes: the capped-round
# contract masks every read by the `live` operand and resolves dead slots
# through the sentinel neutral blend, so the tail is never read as data --
# the old pad's zero tail was deterministic filler, not a consumed value.
# Byte-gated by the runner reference test (which reuses the pool across
# layouts, so stale tails are exercised) and the shard golden. Only the
# concrete (non-traced) capped path pools; the traced whole-layer program is
# untouched.
_LAYER_BUF_POOL: dict[tuple[str, int, Any], Array] = {}


@partial(jax.jit, donate_argnums=(0,))
def _lay_prefix(dst: Array, src: Array) -> Array:
    """Write `src` into `dst`'s prefix in place (`dst` donated: the result
    aliases its memory -- no fresh cap-wide buffer, no tail write). Compiled
    once per (cap, natural width, dtype): the same tiny per-natural-width
    executable class as the eager pads it replaces."""
    return jax.lax.dynamic_update_slice(dst, src, (0,))


def _pool_lay(role: str, src: Array, width: int) -> Array:
    """The pooled, donated form of the concrete capped path's
    `_pad_to_width(src, width, 0)` / `_resize_zero` layer-entry lay-in.
    `role` keys the pool entry: two planes share a width/dtype and each must
    own its buffer -- one shared buffer would be donated twice per layer."""
    if src.shape[0] == width:
        return src
    key = (role, width, src.dtype)
    buf = _LAYER_BUF_POOL.get(key)
    if buf is None:
        buf = jnp.zeros((width,), src.dtype)
    out = _lay_prefix(buf, src)
    _LAYER_BUF_POOL[key] = out
    return out


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
    live_pairs: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Materialized `(s(0), 8*s(1/2), eq mass)` over the stride-2 pairs.

    s(0) reads the even elements at their eq weight; the u=1/2 sum works on
    doubled values (`e0 + e1 = 2*e(1/2)` per factor, likewise eq), which
    `_round_coeffs` rescales. Both go through the shared `LogupSummand` combine
    so the summand cannot drift from the verifier oracle's.

    `live_pairs` masks the sums to the first `live_pairs` pairs -- the
    fixed-width round buffers (xla#179) carry a dead tail past the live state,
    and field addition is exact, so the masked full-width sum is bit-identical
    to the exact-width sum. None sums every pair (the exact-width layout).
    """
    summand = LogupSummand(lam)
    scalars = summand.combine_scalars()
    terms_zero = summand.combine(scalars, eq_0, n0[0::2], d1[0::2], n1[0::2], d0[0::2])
    eq_h = eq_0 + eq_1
    terms_half = summand.combine(
        scalars,
        eq_h,
        n0[0::2] + n0[1::2],
        d1[0::2] + d1[1::2],
        n1[0::2] + n1[1::2],
        d0[0::2] + d0[1::2],
    )
    if live_pairs is not None:
        mask = jnp.arange(terms_zero.shape[0]) < live_pairs
        terms_zero = jnp.where(mask, terms_zero, jnp.zeros_like(terms_zero))
        terms_half = jnp.where(mask, terms_half, jnp.zeros_like(terms_half))
        eq_h = jnp.where(mask, eq_h, jnp.zeros_like(eq_h))
    return jnp.sum(terms_zero), jnp.sum(terms_half), jnp.sum(eq_h)


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
    caps: RoundWidthCaps | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """Run one jagged GKR layer's materialized sumcheck.

    `eval_point` is MSB-first over (batch || row) variables; its length
    fixes the virtual row depth `nrv = len(eval_point) - niv`, which may
    exceed what the materialized row counts need -- the extra rounds fold
    saturated all-ones segments against re-padded neutral rows, exactly the
    virtual positions' values. Returns the bound point (MSB-first, i.e. the
    challenges reversed), the advanced transcript, and the proof.

    `caps` selects the fixed-width round layout (xla#179 size-invariance):
    every round then runs at one static operand shape per phase, live prefix
    tracked by the rounds' `live` operand, so one compiled round kernel serves
    every round -- and every layer and shard proved under the same caps.
    Byte-identical to the exact layout.
    """
    niv = layer.num_batch_variables
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_from_counts(
        planes,
        niv,
        layer.row_counts,
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
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


def _prove_jagged_layer_from_counts(
    planes: _Planes,
    niv: int,
    row_counts: tuple[int, ...],
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    challenge_limbs: int,
    caps: RoundWidthCaps | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """One jagged layer's sumcheck from the layer's static `row_counts`.

    Marker v2 (xla#179 device-derived schedule): the per-round re-pad schedule
    is a pure function of `row_counts` + the round index and derives inside
    the claimed kernels (and the decompositions), so the loop carries only the
    tiny i32[nseg] `row_counts` operand plus per-round i32[3] live triples —
    the hundreds-of-MB host-built gather uploads (and their per-warm-pass
    rebuild/staging) are gone, and the whole-layer jit's HLO stays tiny
    without `row_counts` ever entering the jit key (both ride as operands)."""
    nrv = eval_point.shape[0] - niv
    return _prove_jagged_layer_from_ops(
        planes,
        niv,
        _row_counts_operand(row_counts),
        _round_live_meta(row_counts, nrv),
        None if caps is not None else _round_out_pairs(row_counts, nrv),
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
    )


def _prove_jagged_layer_from_ops(
    planes: _Planes,
    niv: int,
    row_counts: Array,
    live: list[Array],
    out_pairs: tuple[int, ...] | None,
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    challenge_limbs: int,
    caps: RoundWidthCaps | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """`_prove_jagged_layer_from_counts` from PREBUILT schedule operands — the
    seam the whole-layer jit zone routes through, so `row_counts` and the live
    triples ride as TRACED operands (never keying the jit) while `out_pairs`
    (the exact layout's static padded widths; None under caps) stays static
    like `niv`/`caps`."""
    nrv = eval_point.shape[0] - niv
    eq_row = _expand_eq_slice(eval_point, niv, row=True)
    eq_int = _expand_eq_slice(eval_point, niv, row=False)
    naturals, inv_vand = _round_interp_constants(eval_point.dtype)

    state = _JaggedState(planes, eq_row, eq_int, eval_point, lam, claim)
    sched = _JaggedSchedule(
        row_counts,
        live,
        out_pairs,
        _InterpConsts(naturals, inv_vand),
        nrv,
        niv,
        challenge_limbs,
        caps,
    )
    # The host round loop runs one fold-then-compute kernel per round, the FS hop
    # + reduce dispatching between them. `export_dispatch=True` selects the cached
    # per-round `jax.export` binary, but it only fires when this layer runs OUTSIDE
    # an outer jit (`JaggedGkrLayerRound(jit=False)`): the operands are then concrete
    # arrays, so each round host-dispatches and releases its buffers, bounding peak
    # host RAM on wide shards. Under the production outer jit
    # (`JaggedGkrLayerRound(jit=True)`) the dispatch sees tracers and falls back to
    # the marked kernel, tracing the whole loop into one program (the whole-scan
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
    if sched.meta is None:
        raise ValueError(
            "the reference oracle needs the schedule's host-built explicit "
            "meta (_round_metadata) — the round loop's derived-schedule "
            "fields do not carry it"
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
            # The oracle runs the exact layout; the schedule's `live` operand
            # (the fixed-width prefix marker) is the production loop's concern.
            gather, col_index, pair_index, _live = meta[rnd]
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
    """A jagged layer's static round schedule: the layer's `row_counts`
    operand plus the per-round live triples (marker v2 — the re-pad schedule
    itself derives in-kernel from these), the interpolation constants, and
    the batch/row variable counts plus the challenge limb count. Rides beside
    the state so the loop signatures stay `(state, schedule, transcript)`
    rather than a positional-arg bag. `caps` selects the fixed-width round
    layout; None runs the exact (per-round-shape) layout, whose static padded
    widths ride `out_pairs` (None under caps — width-preserving). `meta` (the
    host-built explicit schedule) feeds ONLY the reference oracle
    `_run_jagged_rounds_reference`; the round loop never reads it."""

    row_counts: Array
    live: list[Array]
    out_pairs: tuple[int, ...] | None
    consts: _InterpConsts
    nrv: int
    niv: int
    challenge_limbs: int
    caps: RoundWidthCaps | None = None
    meta: list[tuple[Array | None, Array, Array, Array]] | None = None


@cache
def _row_counts_operand(row_counts: tuple[int, ...]) -> Array:
    """The layer's `row_counts` as the tiny i32[nseg] device operand every v2
    round marker carries (committed once per distinct layout; memoized like
    the live triples)."""
    with jax.ensure_compile_time_eval():
        return jax.device_put(np.asarray(row_counts, np.int32))


@cache
def _dense_live_operand(pairs: int) -> Array:
    """The dense/boundary rounds' i32[2] `{live pairs, 0}` operand, memoized:
    the eager round loop otherwise re-commits this 8-byte array once per round
    per layer -- a real host->device dispatch each time. The value set is tiny
    (one per power-of-two pair count), so the never-evicting cache is safe."""
    with jax.ensure_compile_time_eval():
        return jax.device_put(np.asarray([pairs, 0], np.int32))


def _bind_planes(planes: _Planes, alpha: Array) -> _Planes:
    return _Planes(
        *(_bind_lsb(a, alpha) for a in (planes.n0, planes.n1, planes.d0, planes.d1))
    )


def _round_poly_int(
    planes: _Planes,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_pairs: Array | None = None,
) -> Array:
    """The `sum_as_poly` step for the dense batch phase: the round
    univariate from the current state, no fold (the entry kernel of a
    round loop, before any challenge is bound).

    Fiat-Shamir-less by construction — the FS hop lives in `_fs_reduce`, appended
    after the round compute — so the body is pure field arithmetic. `eq_int` is
    sliced stride-2 once inside `_paired_sums`, over an even state. `live_pairs`
    masks the sum to the live pairs of a fixed-width state (see `_paired_sums`)."""
    eval_zero, eval_half, eq_sum = _paired_sums(
        planes.n0,
        planes.n1,
        planes.d0,
        planes.d1,
        eq_int[0::2],
        eq_int[1::2],
        scalars.lam,
        live_pairs=live_pairs,
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
    live_pairs: Array | None = None,
) -> tuple[Array, _Planes]:
    """The row-variable round body shared by the row kernels: re-pad the four
    MLEs to the round's even layout (`gather`), look the per-pair batch eq
    weight up via `eq_int[col_index]`, and form the round univariate over the
    segment-local `eq_row` pairs. Returns `(poly, planes)` — the padded state the
    caller binds next round.

    The schedule (`gather`, `col_index`, `pair_index`) is host-built; the post-pad
    state is `gather`'s length (even), so the `_paired_sums` stride-2 stays valid.
    `live_pairs` masks the sum to the live pairs of a fixed-width schedule (see
    `_paired_sums`)."""
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
        live_pairs=live_pairs,
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


# --- zorch#327: FS-less compute-only round composites ------------------------
# The host loop wraps each round's fold+sum in a `zorch.sumcheck.round` marker;
# Fiat-Shamir stays the separate `zorch.poseidon2` composite the transcript emits
# between rounds. When no emitter claims the marker (CPU, or a pre-#327 pin), the
# `lax.composite` decomposition runs inline, so the marked path is byte-identical
# to the eager body. All five round bodies are marked: first (round 0, jagged),
# mid row (jagged), boundary (the row->interaction handoff), mid interaction
# (dense), and final (the `_fix_last` fold) -- selected by `_run_jagged_rounds`
# / `_finalize_layer` on both the traced and export-dispatch routes.


def _round_composite_dense_decomp(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    **_attrs: object,
) -> tuple[Array, _Planes, Array]:
    """The `zorch.sumcheck.round` decomposition for the `dense` (interaction) `mid`
    phase -- the byte-exact fallback a recognizing emitter replaces. `_attrs`
    (phase / variant / degree / poly_form) are composite metadata the emitter
    parses; the decomposition needs only the operands. The interp constants ride
    as operands (`naturals` / `inv_vand`) rather than a lifted closure -- the
    emitter may instead rebuild them from `degree` + `poly_form` and drop the two
    trailing operands.

    Width-preserving (xla#179 size-invariance): the state enters live to
    `4 * live[0]` elements (`live[0]` = the round's live reduce pairs) in
    width-`m` buffers and leaves live to `2 * live[0]` in the SAME width -- the
    fold's halved output zero-pads back to `m`, and the sum masks to the live
    pairs. One operand/result shape therefore serves every round of the phase;
    on the exact layout (`4 * live[0] == m`) the live prefix is the whole
    buffer and the values match `_fix_and_sum_int` bit-for-bit."""
    width = planes.n0.shape[0]
    bound = _bind_planes(planes, alpha)
    eq_bound = _bind_lsb(eq_int, alpha)
    poly = _round_poly_int(
        bound,
        eq_bound,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    out = _Planes(
        *(_pad_to_width(a, width, 0) for a in (bound.n0, bound.n1, bound.d0, bound.d1))
    )
    return poly, out, _pad_to_width(eq_bound, width, 0)


def _composite_fix_and_sum_dense(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=mid, variant=dense) marker
    around the uniform interaction fold+sum -- a batched LogUp-GKR round (the
    hardcoded LogUp combine + the eq_adj/pad_adj virtual-mass correction, not a
    plain product). The signature mirrors `_fix_and_sum_int` (interp consts
    threaded as operands, plus the trailing i32[2] `live` prefix marker) so the
    round loop can select it in place. No `challenge_limbs`: the fold challenge
    `alpha` arrives pre-recomposed as one operand whose dtype carries base vs
    extension. Width-preserving: the folded state returns at the input width,
    live to `2 * live[0]`. Byte-identical to `_fix_and_sum_int` on the live
    prefix whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    return composite(
        _round_composite_dense_decomp,
        planes,
        eq_int,
        alpha,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_row_decomp(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    out_pairs: int | None = None,
    **_attrs: object,
) -> tuple[Array, _Planes, Array]:
    """The `zorch.sumcheck.round` decomposition for the `jagged` (row) `mid` phase
    -- the byte-exact fallback a recognizing emitter replaces. `_attrs` (phase /
    variant / degree / poly_form) are composite metadata the emitter parses; the
    decomposition needs only the operands. Marker v2 (xla#179 device-derived
    schedule): the re-pad schedule derives in-trace from `row_counts` + the
    round index `live[2]` (`_derive_row_schedule` — the claimed kernel runs
    the same derivation in place), so no index array is uploaded per round.
    `out_pairs` is the exact layout's STATIC padded pair count, closed over by
    the wrapper (not an operand); None selects the width-preserving capped
    convention (out width = the plane buffer width).

    Size-invariance (xla#179): the sum masks to the `live[0]` live pairs (the
    re-padded state's live prefix is `2 * live[0]`, the rest of the derived
    gather is sentinel → neutral pad), and the folded `eq_row` zero-pads back
    to its input width, live to `live[1] // 2`. On the exact layout the live
    prefixes are the whole buffers and the values match `_fix_and_sum_row`
    bit-for-bit (modulo the eq_row width restore)."""
    eq_width = eq_row.shape[0]
    folded_len = planes.n0.shape[0] // 2
    num_pairs = folded_len if out_pairs is None else out_pairs
    gather, col_index, pair_index = _derive_row_schedule(
        row_counts, live[2], num_pairs, sentinel=folded_len, idx_dtype=jnp.int32
    )
    bound = _bind_planes(planes, alpha)
    eq_bound = _bind_lsb(eq_row, alpha)
    poly, padded = _round_poly_row(
        bound,
        gather,
        col_index,
        pair_index,
        eq_bound,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    return poly, padded, _pad_to_width(eq_bound, eq_width, 0)


def _composite_fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=mid, variant=jagged) marker
    around the row-variable fold+sum -- the segment-based jagged round (the
    hardcoded LogUp combine over the derived re-pad schedule and the
    segment-local `eq_row`, not a plain product). The signature mirrors
    `_fix_and_sum_row` with the schedule operands replaced by the layer's
    `row_counts` (i32[nseg], marker v2) plus the trailing i32[3]
    `{live pairs, live eq_row, round}` operand. `out_pairs` (static, exact
    layout only) sizes the padded output run; None keeps the capped
    width-preserving convention. `eq_row` returns width-preserved (folded
    live prefix, zero tail). Byte-identical to `_fix_and_sum_row` on the live
    prefixes whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    decomp = partial(_round_composite_row_decomp, out_pairs=out_pairs)
    return composite(
        decomp,
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="jagged",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_final_decomp(
    planes: _Planes,
    alpha: Array,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array]:
    """The `zorch.sumcheck.round` decomposition for the `final` phase -- the
    byte-exact fallback a recognizing emitter replaces. Fold only: bind the
    final challenge and read off the four pair openings. No round poly, no eq,
    no interp constants -- nothing sums, so the marker carries just the
    length-2 planes and `alpha`."""
    return _fix_last(planes, alpha)


def _composite_fix_last(
    planes: _Planes, alpha: Array
) -> tuple[Array, Array, Array, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=final, variant=dense)
    marker around the layer tail's final fold (`_fix_last`): bind the last
    challenge and emit the four scalar pair openings. The signature mirrors
    `_fix_last` so `_finalize_layer` can select it in place. Byte-identical to
    `_fix_last` whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    return composite(
        _round_composite_final_decomp,
        planes,
        alpha,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="final",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_boundary_decomp(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    **_attrs: object,
) -> tuple[Array, _Planes]:
    """The `zorch.sumcheck.round` decomposition for the `boundary` (row ->
    interaction handoff) phase -- the byte-exact fallback a recognizing emitter
    replaces. The operand layout is the dense `mid` ABI, but `eq_int` enters at
    2x the post-bind state width and is NOT folded: only the planes bind by
    `alpha`. `eq_int` is dropped from the outputs -- it rides through the round
    unchanged, so emitting it would copy a full tensor per boundary round; the
    wrapper returns the caller's own `eq_int` instead.

    Unlike the width-preserving dense `mid`, the boundary keeps its halving
    output (planes leave at half the input width, live to `2 * live[0]`): it
    fires once per layer, and the halved width is what keeps the plane/eq
    widths equal through the dense phase on the exact layout. The fixed-width
    route slices the output down to its interaction cap outside the marker."""
    bound = _bind_planes(planes, alpha)
    poly = _round_poly_int(
        bound,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    return poly, bound


def _composite_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=boundary, variant=dense)
    marker around the row->interaction handoff -- bind the last row variable's
    challenge (row-shaped fold), then the first interaction round's univariate
    over the still-unfolded `eq_int`. The signature mirrors
    `_fix_and_sum_boundary` (plus the trailing i32[2] `live` prefix marker) so
    the round loop can select it in place; the unchanged `eq_int` is returned
    from outside the marker (not a composite output). Byte-identical to
    `_fix_and_sum_boundary` on the live prefix whenever the marker is unclaimed
    (`lax.composite` runs the decomposition)."""
    poly, planes = composite(
        _round_composite_boundary_decomp,
        planes,
        eq_int,
        alpha,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="boundary",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )
    return poly, planes, eq_int


def _round_composite_first_row_decomp(
    planes: _Planes,
    eq_row: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    out_pairs: int | None = None,
    **_attrs: object,
) -> tuple[Array, _Planes]:
    """The `zorch.sumcheck.round` decomposition for the `jagged` (row) `first`
    phase -- the byte-exact fallback a recognizing emitter replaces. The operand
    order is the `mid` row ABI minus `alpha`: round 0 binds nothing, so there is
    no previous challenge to fold by and the marker carries no `alpha` slot.
    The re-pad schedule derives in-trace from `row_counts` + `live[2]` (= 0,
    marker v2 — see the row `mid` decomposition); the derived indexes span the
    FULL raw height (nothing folds this round). The sum masks to the `live[0]`
    live pairs of a fixed-width schedule (the re-padded state's live prefix is
    `2 * live[0]`)."""
    raw_len = planes.n0.shape[0]
    num_pairs = raw_len // 2 if out_pairs is None else out_pairs
    gather, col_index, pair_index = _derive_row_schedule(
        row_counts, live[2], num_pairs, sentinel=raw_len, idx_dtype=jnp.int32
    )
    return _round_poly_row(
        planes,
        gather,
        col_index,
        pair_index,
        eq_row,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )


def _composite_sum_as_poly_row(
    planes: _Planes,
    row_counts: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=first, variant=jagged)
    marker around the round-0 sum -- no fold, no challenge, just the row-shaped
    round poly over the raw layer plus the re-padded state the caller binds next
    round. The signature mirrors `_round_poly_row` with the schedule operands
    replaced by the layer's `row_counts` (marker v2) plus the trailing i32[3]
    `{live pairs, live eq_row, round}` operand, so the round loop can select it
    in the `sum0` slot. `out_pairs` (static, exact layout only) sizes the
    padded output run; None keeps the capped width-preserving convention.
    Byte-identical to `_round_poly_row` on the live prefixes whenever the
    marker is unclaimed (`lax.composite` runs the decomposition)."""
    decomp = partial(_round_composite_first_row_decomp, out_pairs=out_pairs)
    return composite(
        decomp,
        planes,
        eq_row,
        row_counts,
        eq_int,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="first",
        variant="jagged",
        degree=_DEGREE,
        poly_form="coefficient",
    )


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


def _round_get(key: tuple, *, disk: bool = True) -> export.Exported | None:
    exp = _ROUND_KERNEL_CACHE.get(key)
    if exp is None and disk and _EXPORT_CACHE_DIR is not None:
        path = _export_path(key)
        if path.exists():
            exp = export.deserialize(bytearray(path.read_bytes()))
            _ROUND_KERNEL_CACHE[key] = exp
    return exp


def _round_put(key: tuple, exp: export.Exported, *, disk: bool = True) -> None:
    _ROUND_KERNEL_CACHE[key] = exp
    if disk and _EXPORT_CACHE_DIR is not None:
        # Atomic publish: write a per-pid sibling temp then os.replace into place,
        # so a process sharing ZORCH_EXPORT_CACHE_DIR never deserializes a
        # half-written .bin (rename is atomic within one filesystem).
        path = _export_path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(bytes(exp.serialize()))
        os.replace(tmp, path)


def _round_dispatch(
    key: tuple,
    operands: tuple,
    build: Callable[[], export.Exported],
    *,
    disk: bool = True,
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
    flat operands).

    `disk=False` keeps the binary out of the on-disk cache (in-memory only):
    the multi-round block keys carry the live `Permutation`/fs objects, whose
    default reprs are process-local addresses -- a stable `_export_path` does
    not exist for them, and a colliding address must never alias two configs'
    binaries."""
    exported = _round_get(key, disk=disk)
    if exported is None:
        exported = build()
        _round_put(key, exported, disk=disk)
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
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the dense interaction round through one cached binary symbolic
    over the state size. The round is width-preserving (the folded state
    zero-pads back to the input width, live tracked by `live`), so the state
    and `eq_int` share one `4*g` symbol -- and under fixed caps (xla#179) `g`
    only ever binds one concrete size, so the binary specializes exactly once.

    `exported.call` is a host dispatch; under a `jax.jit` trace the operands are
    tracers, so fall back to the eager kernel -- the jit compiles the round
    itself, the per-round export being its alternative."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_dense(
            planes, eq_int, alpha, scalars, consts, live
        )
    operands = (planes, eq_int, alpha, scalars, live)
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
            jax.ShapeDtypeStruct((2,), live.dtype),
        )
        fn = lambda p, e, al, sc, lv: _composite_fix_and_sum_dense(  # noqa: E731
            p, e, al, sc, consts, lv
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the row->interaction handoff (bind the last row challenge `alpha`,
    then sum the first interaction round over the still-unfolded `eq_int`) through
    one cached binary. Mirrors `_dispatch_fix_and_sum_int` without the `eq_int`
    bind: the bind halves the state (`4*g -> 2*g`) and `eq_int` rides unfolded at
    `2*g` (= the post-bind state), so one dispatched kernel replaces the eager one.
    """
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_boundary(
            planes, eq_int, alpha, scalars, consts, live
        )
    operands = (planes, eq_int, alpha, scalars, live)
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
            jax.ShapeDtypeStruct((2,), live.dtype),
        )
        fn = lambda p, e, al, sc, lv: _composite_fix_and_sum_boundary(  # noqa: E731
            p, e, al, sc, consts, lv
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_sum_as_poly_row(
    planes: _Planes,
    row_counts: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes]:
    """Dispatch the round-0 sum (no fold, no challenge) through one cached
    binary. The capped (width-preserving, `out_pairs` None) route exports
    symbolic over the raw height (`2*h2`) and `eq_row` (`2*rr`) — the derived
    schedule's width follows the plane width, so the marker v2 ABI needs no
    schedule symbol; `eq_int` and `row_counts` ride fixed. The exact route's
    padded width (`out_pairs`) is a static shape, so it exports concrete,
    keyed per layout. Mirrors `_dispatch_fix_and_sum_row` without the bind."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_sum_as_poly_row(
            planes, row_counts, eq_row, eq_int, scalars, consts, live, out_pairs
        )
    operands = (planes, row_counts, eq_row, eq_int, scalars, live)
    key = (
        "sum0",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        consts.naturals.shape[0],
        out_pairs,
        None if out_pairs is None else (planes.n0.shape, eq_row.shape),
    )

    def build() -> export.Exported:
        if out_pairs is None:
            h2, rr = export.symbolic_shape(
                "h2, rr",
                constraints=[
                    "h2 >= 1",
                    f"h2 <= {_ROUND_SYM_MAX}",
                    "rr >= 1",
                    f"rr <= {_ROUND_SYM_MAX}",
                ],
            )
            plane_w, eq_w = 2 * h2, 2 * rr
        else:
            plane_w, eq_w = planes.n0.shape[0], eq_row.shape[0]
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((plane_w,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct((eq_w,), eq_row.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((3,), live.dtype),
        )
        fn = lambda pl, rc, er, ei, sc, lv: _composite_sum_as_poly_row(  # noqa: E731
            pl, rc, er, ei, sc, consts, lv, out_pairs
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes, Array]:
    """Dispatch a jagged row round through one cached binary. The capped
    (width-preserving, `out_pairs` None) route exports symbolic over the input
    state (`2*pp`) and the width-preserved `eq_row` (`2*rr`) — the derived
    schedule's width follows the plane width (marker v2), so no schedule
    symbol exists; `eq_int` and `row_counts` ride fixed, and under fixed caps
    every symbol binds one concrete size, so the binary specializes exactly
    once. The exact route's padded width (`out_pairs`) is a static shape, so
    it exports concrete, keyed per layout."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_row(
            planes,
            eq_row,
            alpha,
            row_counts,
            eq_int,
            scalars,
            consts,
            live,
            out_pairs,
        )
    operands = (
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        live,
    )
    key = (
        "row",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        consts.naturals.shape[0],
        out_pairs,
        None if out_pairs is None else (planes.n0.shape, eq_row.shape),
    )

    def build() -> export.Exported:
        if out_pairs is None:
            pp, rr = export.symbolic_shape(
                "pp, rr",
                constraints=[
                    "pp >= 1",
                    f"pp <= {_ROUND_SYM_MAX}",
                    "rr >= 1",
                    f"rr <= {_ROUND_SYM_MAX}",
                ],
            )
            plane_w, eq_w = 2 * pp, 2 * rr
        else:
            plane_w, eq_w = planes.n0.shape[0], eq_row.shape[0]
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((plane_w,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((eq_w,), eq_row.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((3,), live.dtype),
        )
        fn = (  # noqa: E731
            lambda pl, er, al, rc, ei, sc, lv: _composite_fix_and_sum_row(
                pl, er, al, rc, ei, sc, consts, lv, out_pairs
            )
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


# ============================================================================
# Multi-round blocks (xla#179 host-wall): the decoupled prove's ~330 rounds
# each pay one `call_exported` bind (~55us) plus one FS-zone dispatch -- a
# ~200 ms host wall while the GPU is busy well under half that. A block binary
# runs K uniform mid rounds (round compute + `_fs_reduce` FS hop, challenge
# chained in-trace) per bind, dividing the host wall by K and letting XLA fuse
# the inter-round repack glue that separate binaries must materialize. K is
# static (the in-binary loop unrolls), so the greedy ladder below keeps the
# binary census O(1): one binary per (phase, K, dtype-mix) -- never per shard,
# layer, or position. Under the fixed caps every operand shape is
# round-invariant, which is exactly what lets one K-block serve every stretch.
_ROUND_BLOCK_SIZES = (8, 4, 2)


def _row_live_block(live: list[Array], start: int, k: int) -> Array:
    """The (k, 3) stacked live-triple operand for a row block covering rounds
    `[start, start+k)` -- one tiny stack dispatch per block per layer (the
    triples are the memoized `_round_live_meta` arrays)."""
    return jnp.stack(live[start : start + k])


@cache
def _dense_live_block(pairs0: int, k: int) -> Array:
    """The (k, 2) stacked `{live pairs, 0}` operand for a dense block whose
    first round folds `pairs0` pairs -- consecutive dense rounds halve, so row
    i carries `pairs0 >> i`. Values restate `_dense_live_operand`; memoized per
    (pairs0, k) like it."""
    with jax.ensure_compile_time_eval():
        return jax.device_put(
            np.asarray([[pairs0 >> i, 0] for i in range(k)], np.int32)
        )


def _block_fs_key(transcript: DuplexTranscript, challenge_limbs: int) -> tuple:
    """The FS-config part of a block binary's cache key. The block bakes the
    permutation's constants and the fs backend into its trace (the singles
    never did -- their FS ran outside), so the key must carry them.
    `Poseidon2.__eq__/__hash__` are value-based, making the in-memory dict
    exact; these objects have no stable cross-process repr, which is why block
    binaries skip the on-disk cache (`_round_dispatch(disk=False)`)."""
    return (transcript.permutation, transcript.rate, transcript.fs, challenge_limbs)


def _dispatch_row_block(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_block: Array,
    transcript: DuplexTranscript,
    eval_point: Array,
    pos: Array,
    challenge_limbs: int,
) -> tuple[
    Array, Array, _Planes, Array, DuplexTranscript, Array, Array, Array, Array, Array
]:
    """Dispatch `k = live_block.shape[0]` consecutive capped row rounds --
    each a `fix_and_sum_row` marker plus its `_fs_reduce` FS hop, the round
    challenge chained in-trace -- through ONE cached binary. Only the capped
    (width-preserving, `out_pairs is None`) route exists in block form: the
    exact layout changes width per round, so it keeps the single-round path.

    The transcript crosses the boundary as its five `DuplexState` leaves (the
    permutation / rate / fs metadata is baked into the trace and carried in
    the key via `_block_fs_key`). Returns the stacked `(k, DEGREE+1)` round
    polys, the `(k,)` challenges, and the advanced carry
    `(planes, eq_row, transcript, claim, pad_adj, z_cur, pos, last_r)`."""
    k = live_block.shape[0]
    state_ops = _state_leaves(transcript.state)
    operands = (
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        live_block,
        state_ops,
        eval_point,
        pos,
    )
    key = (
        "row_block",
        k,
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        eval_point.shape,
        tuple(leaf.shape for leaf in state_ops),
        consts.naturals.shape[0],
        consts.naturals.dtype,
        _block_fs_key(transcript, challenge_limbs),
    )

    def build() -> export.Exported:
        pp, rr = export.symbolic_shape(
            "pp, rr",
            constraints=[
                "pp >= 1",
                f"pp <= {_ROUND_SYM_MAX}",
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
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct(live_block.shape, live_block.dtype),
            tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in state_ops),
            jax.ShapeDtypeStruct(eval_point.shape, eval_point.dtype),
            jax.ShapeDtypeStruct(pos.shape, pos.dtype),
        )
        template = transcript

        def fn(
            pl: _Planes,
            er: Array,
            al: Array,
            rc: Array,
            ei: Array,
            sc: _RoundScalars,
            lv: Array,
            st: tuple[Array, Array, Array, Array, Array],
            ep: Array,
            po: Array,
        ) -> tuple[
            Array,
            Array,
            _Planes,
            Array,
            tuple[Array, Array, Array, Array, Array],
            Array,
            Array,
            Array,
            Array,
            Array,
        ]:
            t = replace(template, state=DuplexState(*st))
            dtype = sc.claim.dtype
            pad_adj, z_cur, claim = sc.pad_adj, sc.z_cur, sc.claim
            prev = al
            polys: list[Array] = []
            rs: list[Array] = []
            # eq_adj / lam are row-stretch constants (the eq_adj swap happens
            # at the row->boundary handoff, outside any block), so each
            # iteration rebuilds the scalars bundle around the moving trio.
            for i in range(k):
                sci = _RoundScalars(sc.eq_adj, pad_adj, z_cur, claim, sc.lam)
                poly, pl, er = _composite_fix_and_sum_row(
                    pl, er, prev, rc, ei, sci, consts, lv[i], None
                )
                t, r, claim, pad_adj, z_cur, po = _fs_reduce(
                    poly, t, pad_adj, z_cur, ep, po, challenge_limbs, dtype
                )
                polys.append(poly)
                rs.append(r)
                prev = r
            return (
                jnp.stack(polys),
                jnp.stack(rs),
                pl,
                er,
                _state_leaves(t.state),
                claim,
                pad_adj,
                z_cur,
                po,
                prev,
            )

        return export.export(jax.jit(fn))(*abst)

    out = _round_dispatch(key, operands, build, disk=False)
    polys, rs, planes, eq_row, st, claim, pad_adj, z_cur, pos, prev = out
    return (
        polys,
        rs,
        planes,
        eq_row,
        replace(transcript, state=DuplexState(*st)),
        claim,
        pad_adj,
        z_cur,
        pos,
        prev,
    )


def _dispatch_int_block(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_block: Array,
    transcript: DuplexTranscript,
    eval_point: Array,
    pos: Array,
    challenge_limbs: int,
) -> tuple[
    Array, Array, _Planes, Array, DuplexTranscript, Array, Array, Array, Array, Array
]:
    """`_dispatch_row_block` for `k` consecutive capped dense interaction
    rounds (`fix_and_sum_int` + `_fs_reduce` each, challenge chained
    in-trace). The state and `eq_int` share the dense rounds' `4*g` symbol
    exactly like the single-round dispatch."""
    k = live_block.shape[0]
    state_ops = _state_leaves(transcript.state)
    operands = (planes, eq_int, alpha, scalars, live_block, state_ops, eval_point, pos)
    key = (
        "int_block",
        k,
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        eval_point.shape,
        tuple(leaf.shape for leaf in state_ops),
        consts.naturals.shape[0],
        consts.naturals.dtype,
        _block_fs_key(transcript, challenge_limbs),
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
            jax.ShapeDtypeStruct(live_block.shape, live_block.dtype),
            tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in state_ops),
            jax.ShapeDtypeStruct(eval_point.shape, eval_point.dtype),
            jax.ShapeDtypeStruct(pos.shape, pos.dtype),
        )
        template = transcript

        def fn(
            pl: _Planes,
            ei: Array,
            al: Array,
            sc: _RoundScalars,
            lv: Array,
            st: tuple[Array, Array, Array, Array, Array],
            ep: Array,
            po: Array,
        ) -> tuple[
            Array,
            Array,
            _Planes,
            Array,
            tuple[Array, Array, Array, Array, Array],
            Array,
            Array,
            Array,
            Array,
            Array,
        ]:
            t = replace(template, state=DuplexState(*st))
            dtype = sc.claim.dtype
            pad_adj, z_cur, claim = sc.pad_adj, sc.z_cur, sc.claim
            prev = al
            polys: list[Array] = []
            rs: list[Array] = []
            for i in range(k):
                sci = _RoundScalars(sc.eq_adj, pad_adj, z_cur, claim, sc.lam)
                poly, pl, ei = _composite_fix_and_sum_dense(
                    pl, ei, prev, sci, consts, lv[i]
                )
                t, r, claim, pad_adj, z_cur, po = _fs_reduce(
                    poly, t, pad_adj, z_cur, ep, po, challenge_limbs, dtype
                )
                polys.append(poly)
                rs.append(r)
                prev = r
            return (
                jnp.stack(polys),
                jnp.stack(rs),
                pl,
                ei,
                _state_leaves(t.state),
                claim,
                pad_adj,
                z_cur,
                po,
                prev,
            )

        return export.export(jax.jit(fn))(*abst)

    out = _round_dispatch(key, operands, build, disk=False)
    polys, rs, planes, eq_int, st, claim, pad_adj, z_cur, pos, prev = out
    return (
        polys,
        rs,
        planes,
        eq_int,
        replace(transcript, state=DuplexState(*st)),
        claim,
        pad_adj,
        z_cur,
        pos,
        prev,
    )


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


# Fixed eval_point width for the FS-hop zone below: the recognizer bounds
# num_vars at 62, so 64 covers every layer, and one padded width keeps the
# zone's compile key layer-invariant (the pad tail is never read — the
# dynamic_index rides `pos`, which always points into the live prefix).
_FS_EVAL_POINT_CAP = 64

# The export path's per-round FS hop, hoisted into a module-level jit zone (the
# `_composite.py`-recommended pattern, mirroring poseidon2's `_permute_body`):
# called eagerly between round binaries, the bare `_fs_reduce` re-traces the
# `zorch.duplex_fs` composite's Python body EVERY round and enqueues
# `_reduce_body`'s ~15 element ops one dispatch at a time — measured as the
# dominant host wall of the warm decoupled prove (~330 rounds/shard). The zone
# collapses each hop to one cached-executable dispatch. Keyed by operand
# shapes (eval_point length varies per layer) plus the static squeeze count and
# dtype. jit is byte-transparent, so the transcript stream is unchanged.
_fs_reduce_zone = partial(jax.jit, static_argnums=(6, 7))(_fs_reduce)


def _fs_reduce_dispatch(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    eval_point: Array,
    pos: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array, Array, Array]:
    """Route the FS hop through the jit zone on the eager/export path; under an
    outer trace call the plain body so it keeps fusing into the whole-layer
    program exactly as before (the zone would inline there anyway, but staying
    out preserves the traced path's structure byte-for-byte by construction)."""
    if isinstance(poly, jax.core.Tracer):
        return _fs_reduce(poly, transcript, pad_adj, z_cur, eval_point, pos, n, dtype)
    return _fs_reduce_zone(poly, transcript, pad_adj, z_cur, eval_point, pos, n, dtype)


# The layer tail: the final fold (`_fix_last`) plus stacking the per-round
# challenge/poly lists. Folding `_fix_last` in here keeps the final fold in the
# whole-layer kernel without decorating the bare helper. The width-preserving
# round buffers leave the fully-folded state as the live length-2 prefix, so
# the tail slices it down before the final marker -- the final ABI stays the
# exact (2,) planes.
def _finalize_layer(
    planes: _Planes, alpha: Array, chal: list[Array], poly: list[Array]
) -> tuple[Array, Array, Array, Array, Array, Array]:
    head = _Planes(*(a[:2] for a in (planes.n0, planes.n1, planes.d0, planes.d1)))
    fn0, fn1, fd0, fd1 = _composite_fix_last(head, alpha)
    if all(c.ndim == 0 for c in chal):
        # Per-round entries only (the traced and single-round paths): the
        # original one-stack structure, byte-for-byte.
        return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)
    # Multi-round block segments ((k,) challenges / (k, DEGREE+1) polys) mixed
    # with singles: flatten in round order -- the challenge reversal composes
    # segment reversal with an in-segment flip. Same elements the stacked form
    # carries, concatenated instead of stacked.
    rev = [c[::-1] if c.ndim else c[None] for c in reversed(chal)]
    rows = [p if p.ndim == 2 else p[None] for p in poly]
    return (
        fn0,
        fn1,
        fd0,
        fd1,
        rev[0] if len(rev) == 1 else jnp.concatenate(rev),
        rows[0] if len(rows) == 1 else jnp.concatenate(rows),
    )


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
    peak host RAM (the decoupled production path). Under the outer jit
    (`JaggedGkrLayerRound(jit=True)`) the dispatch sees tracers and falls back
    to the marked kernel, tracing the whole loop into one program. Both paths are
    byte-identical to the inline reference oracle in the tests (same math; the
    export path only regroups it across per-round host dispatches)."""
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    nrv, niv = sched.nrv, sched.niv
    row_counts = sched.row_counts
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    planes = state.planes
    consts = sched.consts
    transcript = cast(DuplexTranscript, transcript)

    # Fixed-width layout (xla#179): lay the state into the capped buffers once
    # at layer entry; every round then runs at one static shape per phase with
    # the live prefix riding the `live` operand, so one compiled round kernel
    # serves every round/layer/shard under the caps. The dead tails are zeros
    # here and never read (every consumer masks by `live`).
    caps = sched.caps
    if caps is not None:
        if caps.row % 4:
            raise ValueError(
                f"row cap {caps.row} must be a multiple of 4 (the boundary "
                "handoff binds then pairs the row-width state, two stride-2 "
                "halvings)"
            )
        if caps.eq_row % 2 or caps.interaction % 4:
            raise ValueError(
                f"eq_row cap {caps.eq_row} must be even and interaction cap "
                f"{caps.interaction} a multiple of 4 (each folds stride-2 "
                "through its rounds)"
            )
        if caps.eq_row < eq_row.shape[0]:
            raise ValueError(
                f"eq_row cap {caps.eq_row} cannot hold the layer's row-eq "
                f"table ({eq_row.shape[0]})"
            )
        if caps.interaction < eq_int.shape[0]:
            raise ValueError(
                f"interaction cap {caps.interaction} cannot hold the layer's "
                f"interaction-eq table ({eq_int.shape[0]})"
            )
        if not isinstance(planes.n0, jax.core.Tracer):
            # Concrete (decoupled) path: lay each layer into the pooled,
            # donated cap buffers -- prefix-only in-place writes instead of
            # fresh cap-wide materializations (see _LAYER_BUF_POOL).
            planes = _Planes(
                *(
                    _pool_lay(f, getattr(planes, f), caps.row)
                    for f in ("n0", "n1", "d0", "d1")
                )
            )
            eq_row = _pool_lay("eq_row", eq_row, caps.eq_row)
            eq_int = _pool_lay("eq_int", eq_int, caps.interaction)
        else:
            planes = _Planes(
                *(
                    _pad_to_width(a, caps.row, 0)
                    for a in (planes.n0, planes.n1, planes.d0, planes.d1)
                )
            )
            eq_row = _pad_to_width(eq_row, caps.eq_row, 0)
            eq_int = _pad_to_width(eq_int, caps.interaction, 0)
    # The dispatch and marked kernels share signatures, so select one per round.
    # Both routes emit the `zorch.sumcheck.round` marker (the dispatch inside its
    # exported binary): a recognizing emitter fuses each round, and an unclaimed
    # marker decomposes inline, byte-identical to the eager body.
    fix_row = (
        _dispatch_fix_and_sum_row if export_dispatch else _composite_fix_and_sum_row
    )
    fix_int = (
        _dispatch_fix_and_sum_int if export_dispatch else _composite_fix_and_sum_dense
    )
    fix_boundary = (
        _dispatch_fix_and_sum_boundary
        if export_dispatch
        else _composite_fix_and_sum_boundary
    )
    # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
    sum0 = _dispatch_sum_as_poly_row if export_dispatch else _composite_sum_as_poly_row
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
    # On the eager path, pad eval_point to the fixed cap so the FS-hop jit
    # zone keys on ONE shape across every layer of the pyramid (eval_point's
    # length grows per layer; per-layer zone compiles multiplied the cold pass
    # ~6x when measured). `pos` and `z_cur` were derived from the live length
    # above, and the zone's `dynamic_index` only ever reads pos < live —
    # value-identical.
    if (
        not isinstance(eval_point, jax.core.Tracer)
        and eval_point.shape[0] <= _FS_EVAL_POINT_CAP
    ):
        eval_point = _pad_to_width(eval_point, _FS_EVAL_POINT_CAP, 0)
    # Multi-round blocks fire only on the decoupled capped path: concrete
    # operands (outside any outer jit) and fixed-width buffers (`out_pairs is
    # None` exactly when the caps fix every round shape). The traced path
    # keeps the single-round structure -- the whole layer already fuses into
    # one program there, so a block would change nothing but the trace shape.
    block_sizes = (
        _ROUND_BLOCK_SIZES
        if export_dispatch
        and caps is not None
        and sched.out_pairs is None
        and not isinstance(planes.n0, jax.core.Tracer)
        else ()
    )
    rnd = 0
    while rnd < nrv + niv:
        if block_sizes and 1 <= rnd < nrv:
            # Greedy row blocks over the uniform mid stretch [1, nrv): K
            # rounds per bind, challenge chained inside the binary.
            k = next((n for n in block_sizes if rnd + n <= nrv), 0)
            if k:
                scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
                (
                    poly,
                    r,
                    planes,
                    eq_row,
                    transcript,
                    claim,
                    pad_adj,
                    z_cur,
                    pos,
                    prev_r,
                ) = _dispatch_row_block(
                    planes,
                    eq_row,
                    prev_r,
                    row_counts,
                    eq_int,
                    scalars,
                    consts,
                    _row_live_block(sched.live, rnd, k),
                    transcript,
                    eval_point,
                    pos,
                    challenge_limbs,
                )
                polys.append(poly)
                challenges.append(r)
                if rnd + k == nrv:
                    # The block covered round nrv-1: the row->boundary swap
                    # (eq_adj takes the row stretch's pad mass) applies here,
                    # exactly as the single-round tail below does it.
                    eq_adj = pad_adj
                    pad_adj = one
                rnd += k
                continue
        if block_sizes and nrv < rnd:
            # Greedy dense blocks over (nrv, nrv+niv): the first covered
            # round folds `1 << (niv - 1 - (rnd - nrv))` pairs, halving per
            # round inside the block.
            k = next((n for n in block_sizes if rnd + n <= nrv + niv), 0)
            if k:
                scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
                (
                    poly,
                    r,
                    planes,
                    eq_int,
                    transcript,
                    claim,
                    pad_adj,
                    z_cur,
                    pos,
                    prev_r,
                ) = _dispatch_int_block(
                    planes,
                    eq_int,
                    prev_r,
                    scalars,
                    consts,
                    _dense_live_block(1 << (niv - 1 - (rnd - nrv)), k),
                    transcript,
                    eval_point,
                    pos,
                    challenge_limbs,
                )
                polys.append(poly)
                challenges.append(r)
                rnd += k
                continue
        scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
        dtype = claim.dtype
        if rnd == 0:
            out_pairs = None if sched.out_pairs is None else sched.out_pairs[0]
            poly, planes = sum0(
                planes,
                row_counts,
                eq_row,
                eq_int,
                scalars,
                consts,
                sched.live[0],
                out_pairs,
            )
        elif rnd < nrv:
            out_pairs = None if sched.out_pairs is None else sched.out_pairs[rnd]
            poly, planes, eq_row = fix_row(
                planes,
                eq_row,
                prev_r,
                row_counts,
                eq_int,
                scalars,
                consts,
                sched.live[rnd],
                out_pairs,
            )
        elif rnd == nrv:
            # The handoff's live pairs: the row phase saturates every segment
            # to one row by construction (row counts <= 2^nrv), so the last
            # padded row layout is exactly two slots per interaction --
            # 2^(niv+1) live elements, 2^(niv-1) post-bind pairs.
            live = _dense_live_operand(1 << (niv - 1))
            # The boundary marker needs its eq operand at half its plane width
            # (the post-bind state), so the capped route reads a resized copy
            # -- the live 2^niv prefix always fits in row // 2 (the last row
            # layout, 2^(niv+1) slots, fits in the row cap). `eq_int` itself
            # rides through the handoff unchanged, at its own width, for the
            # interaction rounds below.
            if caps is None:
                eq_boundary = eq_int
            elif isinstance(eq_int, jax.core.Tracer):
                eq_boundary = _resize_zero(eq_int, caps.row // 2)
            else:
                # Pooled lay-in of the live prefix (concrete path): the old
                # `_resize_zero` wrote a fresh caps.row//2 buffer that is
                # mostly zero tail at a wide cap.
                half = caps.row // 2
                src = eq_int if eq_int.shape[0] <= half else eq_int[:half]
                eq_boundary = _pool_lay("eq_boundary", src, half)
            poly, planes, _ = fix_boundary(
                planes, eq_boundary, prev_r, scalars, consts, live
            )
            if caps is not None:
                # The handoff halves [row] -> [row // 2]; the dense phase runs
                # at the interaction cap, so resize to it -- the live prefix
                # (2^niv elements <= caps.interaction, validated above via the
                # eq table) always survives.
                planes = _Planes(
                    *(
                        _resize_zero(a, caps.interaction)
                        for a in (planes.n0, planes.n1, planes.d0, planes.d1)
                    )
                )
        else:
            live = _dense_live_operand(1 << (niv - 1 - (rnd - nrv)))
            poly, planes, eq_int = fix_int(
                planes, eq_int, prev_r, scalars, consts, live
            )
        # Device FS hop + reduce -- traced into the whole-layer jit on the default
        # path (one fused region per round), dispatched through the cached
        # `_fs_reduce_zone` between rounds on the export path (one executable
        # per hop instead of a composite retrace + ~15 element dispatches).
        # Slices the next z_cur via the decremented `pos`, riding the fold's
        # dispatch instead of a standalone gather.
        transcript, r, claim, pad_adj, z_cur, pos = _fs_reduce_dispatch(
            poly, transcript, pad_adj, z_cur, eval_point, pos, challenge_limbs, dtype
        )
        polys.append(poly)
        challenges.append(r)
        if rnd == nrv - 1:
            eq_adj = pad_adj
            pad_adj = one
        prev_r = r
        rnd += 1

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
    row_counts: Array,
    live: list[Array],
    out_pairs: tuple[int, ...] | None,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """One jagged GKR layer's carry reduction: sample the batching `lam`, prove
    the layer, observe the openings, and fold the carry with the child selector.

    Takes the planes + batch count + prebuilt schedule operands (not a
    `JaggedGkrLayer`) so the whole-layer jit never keys on `row_counts` and
    never bakes the schedule into the trace. A module-level function (no
    implicit `self`) so the chain can drop a round -- and free its layer --
    the moment it builds the next (the one-live-layer release
    `ChainedJaggedProveTest` pins)."""
    num_eval, den_eval, eval_point = carry
    dtype = num_eval.dtype
    transcript = cast(DuplexTranscript, transcript)
    # The per-layer carry brackets the round loop: sample lam + the batched claim
    # before, absorb the openings + sample + fold the child selector after. All
    # device math, traced into the whole-layer jit.
    transcript, lam, claim = _sample_lam_and_claim(
        transcript, num_eval, den_eval, challenge_limbs, dtype
    )
    point, transcript, proof = _prove_jagged_layer_from_ops(
        planes,
        niv,
        row_counts,
        live,
        out_pairs,
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
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


# Shared by every `JaggedGkrLayerRound(jit=True)`. The schedule operands
# (`row_counts` + the per-round live triples) ride as TRACED operands, not
# static args, so `row_counts` values leave the jit key: it keys only on the
# operand SHAPES plus the static `niv` / `challenge_limbs` / `caps` /
# `out_pairs` (`nrv` is read from `eval_point`'s length inside; `out_pairs` is
# None under caps, so the capped pyramid shares one key). Marker v2 shrank
# these operands from the hundreds-of-MB per-round gather arrays to KBs — the
# schedule now derives in-kernel — but the operand-not-closure rule stands:
# baking per-layer values into the trace would recompile per shard. Two layers
# still recompile when their shape sequence differs, but each compile is cheap
# and persistent-cached -- and under `caps` every layer shares ONE shape
# sequence, so the whole pyramid keys to a single trace. Routing through one
# module-level zone lets freshly built same-shape rounds reuse a single trace,
# so a consumer rebuilding the chain each warm iteration (the generator
# keeping lazy one-live-layer release) re-traces at most per distinct shape
# sequence, not per iter.
@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _jagged_round_zone(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    row_counts: Array,
    live: list[Array],
    niv: int,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    out_pairs: tuple[int, ...] | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    planes = _Planes(numerator_0, numerator_1, denominator_0, denominator_1)
    return _prove_jagged_layer_round(
        planes,
        niv,
        row_counts,
        live,
        out_pairs,
        challenge_limbs,
        caps,
        carry,
        transcript,
    )


def _jagged_round_via_zone(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Build the schedule operands host-side and dispatch through
    `_jagged_round_zone` with the planes + `row_counts` + live triples as
    traced operands. Splitting them out of the trace (rather than closing over
    the layer's static `row_counts`) is what keeps the whole-layer compile
    shard-independent."""
    niv = layer.num_batch_variables
    eval_point = carry[2]
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    return _jagged_round_zone(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        niv,
        challenge_limbs,
        caps,
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
        carry,
        transcript,
    )


def _jagged_round_eager(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """The `jit=False` body: build the schedule operands host-side and run the
    round loop eagerly -- each round (and its FS hop) dispatches on its own, so
    the export path can release every round's buffers before the next (the
    decoupled wide-shard production path)."""
    niv = layer.num_batch_variables
    eval_point = carry[2]
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_round(
        planes,
        niv,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
        challenge_limbs,
        caps,
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

    With `jit=True` the per-layer prove dispatches through the module-level
    `_jagged_round_zone` with the round schedule as a traced operand, so it keys
    on `(niv, plane shapes)` and never on `row_counts` -- the trace (and its
    compiled kernel) is reused across every same-shape round, and shards
    differing only in row counts share one compile. With `jit=False` (default)
    the round loop runs eagerly -- each round's marked kernel (and the export
    dispatch, when it fires) releases its buffers before the next, bounding peak
    host RAM on wide shards. The round holds only its layer (no per-instance
    jit, no self-closure), so the chain's release bound is untouched. The
    pyramid stays a host-orchestrated Python loop of these (one trace per layer
    shape, never one `jit` over the whole pyramid -- it does not fit at scale;
    see `prover.LogupSumcheckRound`).
    """

    def __init__(
        self,
        layer: JaggedGkrLayer,
        challenge_limbs: int = 1,
        *,
        jit: bool = False,
        caps: RoundWidthCaps | None = None,
    ) -> None:
        # `partial` closes over (layer, challenge_limbs, caps), not `self`, so
        # the chain frees the round -- and its layer -- the moment it builds the
        # next. `jit=True` dispatches through the shared module-level zone, so
        # same-shape rounds reuse one trace instead of re-compiling per call.
        # `caps` selects the fixed-width round layout (see `prove_jagged_layer`).
        body = _jagged_round_via_zone if jit else _jagged_round_eager
        self._call = partial(body, layer, challenge_limbs, caps)

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
