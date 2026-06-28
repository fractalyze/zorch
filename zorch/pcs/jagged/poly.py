# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged Little Polynomial — verifier point-eval via branching program.

H(r, c, i) = 1  iff  t_c ≤ i < t_{c+1}  AND  i − t_c = r.
Adapts whir-zorch `jagged/poly.py` to AOT-clean form: static l_max column axis,
lax.fori_loop layer loop, n_d bound to the instance's log-area tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.pcs.jagged.dense import log_area_tier
from zorch.poly.eq import expand_eq_to_hypercube

NUM_MEMORY_STATES = 4  # (carry, comparison_so_far)
NUM_BIT_STATES = 16  # (row_bit, index_bit, t_c_bit, t_{c+1}_bit)


@dataclass(frozen=True)
class _MemoryState:
    carry: bool
    comparison_so_far: bool

    @property
    def index(self) -> int:
        return int(self.carry) + (int(self.comparison_so_far) << 1)


def _all_memory_states() -> list[_MemoryState]:
    return [
        _MemoryState(carry=(c != 0), comparison_so_far=(s != 0))
        for s in range(2)
        for c in range(2)
    ]


@dataclass(frozen=True)
class _BitState:
    row_bit: bool
    index_bit: bool
    curr_prefix_bit: bool
    next_prefix_bit: bool


def _all_bit_states() -> list[_BitState]:
    return [
        _BitState(
            row_bit=(r != 0),
            index_bit=(i != 0),
            curr_prefix_bit=(c != 0),
            next_prefix_bit=(n != 0),
        )
        for r in range(2)
        for i in range(2)
        for c in range(2)
        for n in range(2)
    ]


def _transition(bs: _BitState, ms: _MemoryState) -> _MemoryState | None:
    """δ(bit, memory) → new memory state, or None on addition-check failure."""
    three_sum = int(bs.row_bit) + int(ms.carry) + int(bs.curr_prefix_bit)
    if int(bs.index_bit) != (three_sum & 1):  # i_bit = (r+carry+t_c) mod 2
        return None
    new_carry = (three_sum >> 1) != 0
    # MSB-first comparison: a differing bit (processed later = more significant) wins.
    if bs.index_bit == bs.next_prefix_bit:
        new_cmp = ms.comparison_so_far
    else:
        new_cmp = bs.next_prefix_bit
    return _MemoryState(carry=new_carry, comparison_so_far=new_cmp)


def _build_transition_rows() -> list[list[int]]:
    rows = []
    for ms in _all_memory_states():
        for bs in _all_bit_states():
            row = [0] * NUM_MEMORY_STATES
            out = _transition(bs, ms)
            if out is not None:
                row[out.index] = 1
            rows.append(row)
    return rows  # 64 × 4


_TRANSITION_ROWS = _build_transition_rows()
_SUCCESS_INDEX = _MemoryState(carry=False, comparison_so_far=True).index  # 2
_INITIAL_INDEX = _MemoryState(carry=False, comparison_so_far=False).index  # 0


def bp_eval_core(
    z_row: Array,
    z_index: Array,
    prefix_sum: Array,
    next_prefix_sum: Array,
    t_matrix: Array,
    num_vars: int,
) -> Array:
    """h(z_row, z_index; t_c, t_{c+1}) — 4-state DP fused via lax.fori_loop.

    num_vars is a compile-time constant. The layer loop runs MSB->LSB
    (layer = num_vars->0); each layer: 4 bits -> eq16 -> [4,4] transition
    matrix (vmap) -> sv update.

    NOTE: jnp.tile aborts on ZKX field dtypes with an MLIR bit-width assertion,
    so it is replaced by vmap(lambda tm: eq16 @ tm)(t_res) — mathematically
    identical.
    """
    dtype = z_row.dtype
    r_dim, i_dim = z_row.shape[0], z_index.shape[0]
    p_dim, n_dim = prefix_sum.shape[0], next_prefix_sum.shape[0]
    zero = jnp.zeros([], dtype=dtype)
    # t_res: (NUM_MEMORY_STATES, NUM_BIT_STATES, NUM_MEMORY_STATES)
    t_res = t_matrix.reshape(NUM_MEMORY_STATES, NUM_BIT_STATES, NUM_MEMORY_STATES)

    def _bit(vec: Array, dim: int, layer: Array) -> Array:
        # 0 when layer >= dim (high-bit padding). jnp.clip keeps the OOB index safe.
        idx = jnp.clip(dim - 1 - layer, 0, dim - 1)
        return jnp.where(layer < dim, vec[idx], zero)

    def body(k: Array, sv: Array) -> Array:
        layer = num_vars - 1 - k
        point = jnp.stack(
            [
                _bit(z_row, r_dim, layer),
                _bit(z_index, i_dim, layer),
                _bit(prefix_sum, p_dim, layer),
                _bit(next_prefix_sum, n_dim, layer),
            ]
        )
        eq16 = expand_eq_to_hypercube(point, jnp.ones([], dtype=dtype))  # [16]
        # t_layer[m, m'] = eq16 @ t_res[m]  — vmap over input memory state m
        t_layer = jax.vmap(lambda tm: eq16 @ tm)(t_res)  # [4,4]
        return t_layer @ sv

    sv0 = (
        jnp.zeros(NUM_MEMORY_STATES, dtype=dtype)
        .at[_SUCCESS_INDEX]
        .set(jnp.ones([], dtype=dtype))
    )
    # num_vars layers (not num_vars+1): a layer == num_vars would read all-zero
    # bits (every dim <= num_vars) and is an identity on the carry-0 start
    # state, so it's a no-op — omit it.
    sv = jax.lax.fori_loop(0, num_vars, body, sv0)
    return sv[_INITIAL_INDEX]


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
    """`(l_max+1, n_d)` prefix-sum bit tensor whose canonical int32 limb-0 holds
    each MSB-first bit, typed as `dtype`.

    `partial_eval_core` derives its integer scatter offsets by bitcasting the bit
    tensor to int32 and reading limb 0 — it needs the *canonical* bit there. A
    Montgomery field dtype encodes `astype(1)` as `R mod p` (limb 0 ≠ 1), so
    `build_jagged_layout`'s field-valued tensor (correct for the inner sumcheck's
    field arithmetic) misreads under that raw bitcast. Build a separate tensor
    by packing the bits into int32 limb 0 (other limbs zero) and bitcasting to
    the field dtype — its raw bytes give the right offsets regardless of the
    field's Montgomery-ness. Limb count is derived from the dtype's storage.
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

    Derives the integers on-device via bitcast (no host numpy), shape-polymorphic
    in BOTH axes: ``n_d`` (the prefix-bit width) may be a symbolic export dim.
    EF bitcasts to a trailing limb axis (16B→4 int32), base field does not
    (4B→1 int32); take canonical limb 0 either way via an ndim test, NOT a reshape
    (a symbolic-``n_d`` reshape can't infer the -1 limb dim). The MSB-first weights
    ``2^(n_d-1-k)`` are built with a vectorized ``arange(n_d)`` (``range(n_d)`` is
    a Python loop that can't iterate a symbolic dim)."""
    limbs = jax.lax.bitcast_convert_type(col_prefix_sums, jnp.int32)
    bit_vals = limbs[..., 0] if limbs.ndim > col_prefix_sums.ndim else limbs
    powers = jnp.left_shift(jnp.int32(1), n_d - 1 - jnp.arange(n_d, dtype=jnp.int32))
    return jnp.sum(bit_vals * powers, axis=1)  # (l_max+1,) int32


def _count_leq_sorted(sorted_arr: Array, queries: Array, n_steps: int) -> Array:
    """For each ``q`` in ``queries``: ``#{j : sorted_arr[j] <= q}`` —
    ``searchsorted(side="right")`` over an ascending array.

    A vectorized binary search of a STATIC ``n_steps`` (>= ⌈log2(len+1)⌉), so it is
    both memory-light (no ``(len(queries), len)`` materialization) and symbolic-safe
    in ``len(sorted_arr)``: ``jnp.searchsorted`` bakes that length as a constant,
    which a symbolic column count forbids. Over-provisioned steps are no-ops (once
    ``lo == hi`` the update is idempotent)."""
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
    """J̃(z_row, z_col, ·) over the first ``size`` dense indices — shape-polymorphic
    in the column count ``col_prefix_sums.shape[0]`` AND the prefix-bit width
    ``n_d = col_prefix_sums.shape[1]``.

    ``size`` is the output domain (the caller's dense area). Materializing only
    ``[0, size)`` — rather than the full ``2^n_d`` then slicing — is what lets
    ``n_d`` be a symbolic dim: ``2^n_d`` is exponential in ``n_d`` (not a
    polynomial export dim), whereas ``n_d`` survives ONLY as the decode bit width
    (polynomial). Every nonzero indicator entry lands below ``total_area <= size``,
    so this is byte-identical to the full-domain-then-slice form. See
    ``partial_eval`` for the gather / searchsorted rationale and the limb
    convention.  Output shape: ``(size,)``."""
    dtype = z_row.dtype
    one = jnp.ones([], dtype=dtype)
    n_d = col_prefix_sums.shape[1]
    n_r = z_row.shape[0]
    row_len = jnp.left_shift(jnp.int32(1), n_r)  # 2^n_r, as a value (n_r symbolic)

    prefix_sums_int = _decode_prefix_sums(col_prefix_sums, n_d)

    col_eq = expand_eq_to_hypercube(z_col, one)  # [2^n_c] (n_c static, a table)

    # c_idx = (#prefix entries ≤ i) − 1 — the owning column, last c with t_c ≤ i
    # (searchsorted side="right"; duplicate prefix entries from zero-height columns
    # resolve to the real owner). n_steps = n_c + 1 covers the l_max+1 entries. No
    # clamp on c_idx — the tail i ≥ t_L lands c_idx at the last index; the t_{c+1}
    # gather clamps OOB by default and the height mask zeros it (byte-identical to
    # the clamped form).
    i_idx = jnp.arange(size, dtype=jnp.int32)
    c_idx = _count_leq_sorted(prefix_sums_int, i_idx, z_col.shape[0] + 1) - 1
    t_c = prefix_sums_int[c_idx]
    h = prefix_sums_int[c_idx + 1] - t_c  # column height (0 for padding columns)
    local = i_idx - t_c
    # min(h, row_len): the row eq covers 2^n_r rows, so a taller-than-capacity
    # column truncates — identical to the scatter form's fixed-width window.
    mask = local < jnp.minimum(h, row_len)

    # eq(z_row, local) per element instead of a 2^n_r gather table, so n_r can be
    # symbolic (2^n_r is exponential in n_r). row_eq[j] = ∏_k eq(z_row[k],
    # bit_{n_r-1-k}(j)) — expand_eq_to_hypercube's MSB-first convention (z_row[0] =
    # MSB). A lax.scan over the n_r bits (field multiplies, no EF reduce_prod) so
    # the trip count can be symbolic; the product commutes, so scan order is moot.
    # The mask zeros entries with local >= row capacity, so the low-n_r-bit read
    # (no clamp) is identical to the table's clamped gather.
    def _eq_bit(acc: Array, k: Array) -> tuple[Array, None]:
        bit = ((local >> (n_r - 1 - k)) & 1).astype(dtype)
        z_k = z_row[k]
        return acc * (bit * z_k + (one - bit) * (one - z_k)), None

    row_vals, _ = jax.lax.scan(
        _eq_bit, jnp.ones(i_idx.shape, dtype), jnp.arange(n_r, dtype=jnp.int32)
    )
    val = col_eq[c_idx] * row_vals
    return jnp.where(mask, val, jnp.zeros([], dtype=dtype))
