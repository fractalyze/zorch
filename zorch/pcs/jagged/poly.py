# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged Little Polynomial — layout construction + polymorphic partial-eval.

Builds the column prefix-sum bit tensor (`build_jagged_layout`) and evaluates
the partial MLE J̃(z_row, z_col, ·) over the dense area (`partial_eval_core`),
shape-polymorphic in the column count and prefix-bit width. Adapts whir-zorch
`jagged/poly.py` to AOT-clean form: static l_max column axis, n_d bound to the
instance's log-area tier. The branching-program indicator eval that closes the
sumcheck lives in `branching_program.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.pcs.jagged.dense import log_area_tier
from zorch.poly.eq import expand_eq_to_hypercube


def msb_first_bits(values: Any, num_bits: int) -> np.ndarray:
    """(N,) ints → (N, num_bits) numpy int64, MSB first. Host-side; never feeds a
    field element into >> (ZKX field dtypes have no lax.shift)."""
    arr = np.asarray(values, dtype=np.int64)
    shifts = np.arange(num_bits - 1, -1, -1, dtype=np.int64)
    return (arr[:, None] >> shifts[None, :]) & 1


def build_prefix_sums(row_counts: Sequence[int]) -> list[int]:
    sums = [0]
    for h in row_counts:
        sums.append(sums[-1] + h)
    return sums


def build_jagged_layout(
    row_counts: Sequence[int], l_max: int, dtype: Any
) -> tuple[Array, int]:
    """heights -> (col_prefix_sums (l_max+1, n_d) field bit tensor, n_d).

    ``n_d`` (the BP layer count = log-area tier) is the only static dim downstream
    code needs; every other dim is derived from array shapes by the shape-polymorphic
    cores. Padding columns use an EMPTY RANGE (t_c = t_{c+1} = t_L) — zero-bit
    padding injects a phantom range that corrupts J̃, so it is forbidden.
    """
    real_L = len(row_counts)
    if real_L > l_max:
        raise ValueError(f"real_L={real_L} > l_max={l_max}")
    prefix = build_prefix_sums(row_counts)  # length real_L+1
    n_d = log_area_tier(prefix[-1])
    padded = prefix + [prefix[-1]] * (l_max - real_L)  # length l_max+1, empty-range pad
    cps = jnp.asarray(msb_first_bits(padded, n_d), dtype=dtype)
    return cps, n_d


def _offset_bit_tensor(
    col_heights: list[int], l_max: int, n_d: int, dtype: Any
) -> Array:
    """`(l_max+1, n_d)` MSB-first prefix-sum bit tensor, typed as `dtype`, whose
    raw int32 limb 0 holds each bit canonically.

    `partial_eval_core` reads integer offsets by bitcasting limb 0, so it needs
    the canonical bit there. A Montgomery dtype encodes `astype(1)` as `R mod p`
    (limb 0 ≠ 1), so `build_jagged_layout`'s field-valued tensor misreads under a
    raw bitcast; pack the bits straight into int32 limb 0 (other limbs zero) and
    bitcast, so the bytes decode canonically regardless of Montgomery-ness.
    """
    prefix = build_prefix_sums(col_heights)  # length len+1
    padded = prefix + [prefix[-1]] * (l_max - len(col_heights))  # empty-range pad
    bits = msb_first_bits(padded, n_d)  # (l_max+1, n_d) int
    # int32 limbs per field element (4 for a 128-bit EF, 1 for a 32-bit base).
    probe = jax.lax.bitcast_convert_type(jnp.zeros((1,), dtype), jnp.int32)
    n_limbs = probe.shape[-1] if probe.ndim > 1 else 1
    limbs = np.zeros((bits.shape[0], n_d, n_limbs), dtype=np.int32)
    limbs[..., 0] = bits
    return jax.lax.bitcast_convert_type(jnp.asarray(limbs), dtype)


def _decode_prefix_sums(col_prefix_sums: Array, n_d: Any) -> Array:
    """(l_max+1, n_d) MSB-first bit tensor → (l_max+1,) int32 prefix sums.

    On-device (no host numpy), symbolic-safe in ``n_d``. EF bitcasts to a trailing
    limb axis (16B→4 int32), base field does not (4B→1 int32); read limb 0 via an
    ndim test, not a reshape (symbolic ``n_d`` can't infer the -1 limb dim). Weights
    ``2^(n_d-1-k)`` use a vectorized ``arange(n_d)`` (``range`` can't iterate a
    symbolic dim)."""
    limbs = jax.lax.bitcast_convert_type(col_prefix_sums, jnp.int32)
    bit_vals = limbs[..., 0] if limbs.ndim > col_prefix_sums.ndim else limbs
    powers = jnp.left_shift(jnp.int32(1), n_d - 1 - jnp.arange(n_d, dtype=jnp.int32))
    return jnp.sum(bit_vals * powers, axis=1)  # (l_max+1,) int32


def _count_leq_sorted(sorted_arr: Array, queries: Array, n_steps: int) -> Array:
    """For each ``q`` in ``queries``: ``#{j : sorted_arr[j] <= q}`` —
    ``searchsorted(side="right")`` over an ascending array.

    A vectorized binary search of a STATIC ``n_steps`` (>= ⌈log2(len+1)⌉): memory-
    light and symbolic-safe in ``len(sorted_arr)`` (``jnp.searchsorted`` bakes that
    length as a constant, which a symbolic column count forbids). Over-provisioned
    steps are no-ops (idempotent once ``lo == hi``)."""
    n = sorted_arr.shape[0]
    lo = jnp.zeros(queries.shape, jnp.int32)
    hi = jnp.full(queries.shape, n, jnp.int32)
    for _ in range(n_steps):
        mid = (lo + hi) // 2
        val = sorted_arr[jnp.minimum(mid, n - 1)]  # guard mid == n
        go_right = (mid < n) & (val <= queries)
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(go_right, hi, mid)
    return lo


def partial_eval_core(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    size: Any,
) -> Array:
    """J̃(z_row, z_col, ·) over the first ``size`` dense indices, shape-polymorphic
    in the column count ``col_prefix_sums.shape[0]`` and the prefix-bit width
    ``n_d = col_prefix_sums.shape[1]``. Output shape ``(size,)``.

    Building only ``[0, size)`` (the caller's dense area) rather than the full
    ``2^n_d`` then slicing is what lets ``n_d`` be symbolic: ``2^n_d`` is
    exponential in ``n_d``, but over ``size`` it survives only as the decode bit
    width. Every nonzero indicator entry lands below ``total_area <= size``, so
    this is byte-identical to the full-domain-then-slice form."""
    dtype = z_row.dtype
    one = jnp.ones([], dtype=dtype)
    n_d = col_prefix_sums.shape[1]
    n_r = z_row.shape[0]
    row_len = jnp.left_shift(jnp.int32(1), n_r)  # 2^n_r, as a value (n_r symbolic)

    prefix_sums_int = _decode_prefix_sums(col_prefix_sums, n_d)

    col_eq = expand_eq_to_hypercube(z_col, one)  # [2^n_c] (n_c static, a table)

    # c_idx = owning column = (#prefix entries ≤ i) − 1 (searchsorted side="right";
    # duplicate prefixes from zero-height columns resolve to the real owner). No
    # clamp: the tail i ≥ t_L lands at the last index, where the default OOB gather
    # and the height mask zero it — byte-identical to clamping.
    i_idx = jnp.arange(size, dtype=jnp.int32)
    # n_c+2 steps (not n_c+1): l_max==1 gives n_c==0, where one step under-searches
    # the 2 prefix entries; over-provisioned steps are no-ops.
    c_idx = _count_leq_sorted(prefix_sums_int, i_idx, z_col.shape[0] + 2) - 1
    t_c = prefix_sums_int[c_idx]
    h = prefix_sums_int[c_idx + 1] - t_c  # column height (0 for padding columns)
    local = i_idx - t_c
    # min(h, row_len): the row eq covers 2^n_r rows, so a taller-than-capacity
    # column truncates — identical to the scatter form's fixed-width window.
    mask = local < jnp.minimum(h, row_len)

    # eq(z_row, local) per element instead of a 2^n_r gather table, so n_r can be
    # symbolic (2^n_r is exponential in n_r). row_eq[j] = ∏_k eq(z_row[k],
    # bit_{n_r-1-k}(j)), MSB-first per expand_eq_to_hypercube. A lax.scan over the
    # n_r bits gives a symbolic trip count; the product commutes, so order is moot.
    def _eq_bit(acc: Array, k: Array) -> tuple[Array, None]:
        bit = ((local >> (n_r - 1 - k)) & 1).astype(dtype)
        z_k = z_row[k]
        return acc * (bit * z_k + (one - bit) * (one - z_k)), None

    row_vals, _ = jax.lax.scan(
        _eq_bit, jnp.ones(i_idx.shape, dtype), jnp.arange(n_r, dtype=jnp.int32)
    )
    val = col_eq[c_idx] * row_vals
    return jnp.where(mask, val, jnp.zeros([], dtype=dtype))
