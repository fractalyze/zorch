# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Static round schedules of the jagged sumcheck: pure functions of a
layer's Python-int `row_counts` (fold/re-pad layouts, live triples, block
operands), plus their memoized device commits."""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.logup_gkr.circuit import (
    _prepad_folded,
    _segment_gather_np,
)


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


def _row_live_block(live: list[Array], start: int, k: int) -> Array:
    """The (k, 3) stacked live-triple operand for a row block covering rounds
    `[start, start+k)` -- one tiny stack dispatch per block per layer (the
    triples are the memoized `_round_live_meta` arrays). The fallback for a
    hand-built schedule with no `counts`; the layout-keyed route below is the
    zero-dispatch production path."""
    return jnp.stack(live[start : start + k])


@cache
def _row_live_blocks(
    row_counts: tuple[int, ...], num_row_vars: int, start: int, k: int
) -> Array:
    """`_row_live_block` pre-staged per layout: the greedy block partition is
    a pure function of the layout, so the (k, 3) stacks commit once per
    (layout, block) into this never-evicting cache (tiny i32 rows, the same
    policy as `_round_live_meta`) instead of one stack dispatch per block per
    layer per pass."""
    with jax.ensure_compile_time_eval():
        return jnp.stack(_round_live_meta(row_counts, num_row_vars)[start : start + k])


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
