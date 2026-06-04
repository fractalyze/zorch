# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged Little Polynomial — verifier point-eval via branching program.

H(r, c, i) = 1  iff  t_c ≤ i < t_{c+1}  AND  i − t_c = r.
Adapts whir-zorch `jagged/poly.py` to AOT-clean form: static l_max column axis,
lax.fori_loop layer loop, n_d bound to the instance's log-area tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from functools import reduce as _reduce
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.utils.bits import log2_ceil_usize

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


@dataclass(frozen=True)
class JaggedStaticConfig:
    """Static key for eval_jagged_mle. frozen -> hashes by field (jit static_argnames).

    l_max: compiled-max column count.  n_c = ceil(log2 l_max).
    n_r:   row-bit width (capacity ceiling).  n_d: BP layer count (this
           instance's log-area tier).
    dtype: field dtype (field-agnostic).
    """

    l_max: int
    n_c: int
    n_r: int
    n_d: int
    dtype: object


def build_jagged_layout(
    row_counts: Sequence[int], l_max: int, n_r: int, dtype: Any
) -> tuple[Array, JaggedStaticConfig]:
    """heights -> (col_prefix_sums (l_max+1, n_d) field bit tensor, JaggedStaticConfig).

    Padding columns use an EMPTY RANGE (t_c = t_{c+1} = t_L) — zero-bit padding
    injects a phantom range that corrupts J̃, so it is forbidden.
    """
    real_L = len(row_counts)
    if real_L > l_max:
        raise ValueError(f"real_L={real_L} > l_max={l_max}")
    prefix = build_prefix_sums(row_counts)  # length real_L+1
    total_area = prefix[-1]
    n_d = log2_ceil_usize(total_area) + 1
    if total_area >= (1 << n_d):
        raise ValueError(f"total_area={total_area} >= 2^{n_d}")
    n_c = log2_ceil_usize(l_max)
    padded = prefix + [prefix[-1]] * (l_max - real_L)  # length l_max+1, empty-range pad
    cps = jnp.asarray(msb_first_bits(padded, n_d), dtype=dtype)
    return cps, JaggedStaticConfig(l_max=l_max, n_c=n_c, n_r=n_r, n_d=n_d, dtype=dtype)


@partial(jax.jit, static_argnames=("cfg",))
def eval_jagged_mle(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_index: Array,
    *,
    cfg: JaggedStaticConfig,
) -> Array:
    """J̃(z_row, z_col, z_index) = Σ_c eq(z_col, c) · h(z_row, z_index; t_c, t_{c+1}).

    col_prefix_sums: (l_max+1, n_d) field bit tensor.  z_*: (n_r/n_c/n_d,) challenges.
    Output shape () — a function of input SHAPES (no value dependence). The
    column vmap is over the static l_max.

    NOTE: jnp.sum(arr) aborts on the ZKX backend — EF (koalabearx4) reduce_sum is
    unsupported and hits an MLIR assertion. Since cfg.l_max is static, sum via a
    Python-level functools.reduce, unrolled at trace time (same semantics,
    AOT-clean). This unroll grows the trace linearly in l_max — fine for the
    verifier-once / small-l_max case here, but revert to a single reduction once
    EF reduce_sum is supported.
    """
    dtype = z_row.dtype
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    num_vars = max(cfg.n_r, cfg.n_d)
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones([], dtype=dtype))  # 2^{n_c}
    all_left = col_prefix_sums[: cfg.l_max]
    all_right = col_prefix_sums[1:]
    bp_evals = jax.vmap(
        lambda pl, pr: bp_eval_core(z_row, z_index, pl, pr, t_matrix, num_vars),
        in_axes=(0, 0),
    )(all_left, all_right)
    # Unroll sum at trace time (l_max is static) — avoids EF reduce_sum crash.
    return _reduce(
        lambda a, b: a + b, [col_eq[c] * bp_evals[c] for c in range(cfg.l_max)]
    )
