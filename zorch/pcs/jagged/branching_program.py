# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged indicator via a 4-state branching program.

The MLE of the jagged indicator ``H(r, c, i) = 1 iff i = t_c + r and
t_c <= i < t_{c+1}`` evaluated at the sumcheck's random point, via a 4-state
(carry, comparison) automaton folded over the prefix bits MSB->LSB — the
"un-jagging" glue that reduces the ragged trace to a dense sumcheck.

``bp_eval_core`` is the per-column fold, wrapped in the ``zorch.jagged_bp``
name-routed composite so a vendor fuses the whole DP into one register-resident
kernel. The branching program exists only for this indicator (no other caller),
so the marker is domain-named rather than a generic ``matrix_fold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx
import frx.numpy as jnp
from frx import Array

from zorch.fusion import fused_region
from zorch.poly.eq import expand_eq_to_hypercube

NUM_MEMORY_STATES = 4  # (carry, comparison_so_far)
NUM_BIT_STATES = 16  # (row_bit, index_bit, t_c_bit, t_{c+1}_bit)

# Name-routed composite for the per-column BP indicator fold: a vendor fuses the
# whole `fori_loop` DP (a 4-vector folded through num_vars soft 4×4 transitions)
# into one register-resident kernel, replacing the ~thousands of launch-bound
# tiny extension-field matmuls the batched form scatters. Domain-named — the
# branching program exists only for the jagged indicator, so a generic
# `matrix_fold` name would be YAGNI (see fractalyze/zorch#413). The decomposition
# (`_bp_eval_decomposition`) is byte-identical to the pre-marker body, so an
# unrecognized marker inlines it with no behavior change.
JAGGED_BP_MARKER = "zorch.jagged_bp"
# Version rides as `composite.version`; producer and the xla recognizer are
# pinned together, so it moves only on a cross-release ABI break.
JAGGED_BP_MARKER_VERSION = 1


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


def _bp_eval_decomposition(
    z_row: Array,
    z_index: Array,
    prefix_sum: Array,
    next_prefix_sum: Array,
    t_matrix: Array,
    **_attrs: object,
) -> Array:
    """The `zorch.jagged_bp` decomposition: the 4-state DP over the BP layers via
    lax.fori_loop, MSB->LSB. Runs verbatim when no vendor emits the marker (an
    unrecognized composite inlines its decomposition), so it is byte-identical to
    the pre-marker body. Each layer: 4 bits -> eq16 -> [4,4] transition matrix ->
    state-vector update.

    The layer count ``num_vars = max(n_r, n_d)`` is derived from the operand
    shapes rather than passed as an operand (a bare scalar operand is
    constant-sunk into the custom fusion by XLA, shifting the emitter's positional
    ABI). ``n_d`` is the PREFIX width (``prefix_sum.shape[0]``), NOT
    ``z_index.shape[0]`` — ``z_index`` is ``z_trace``, which is ``2*n_d`` wide, so
    conflating them runs the fold for the wrong layer count. Deriving keeps a
    symbolic export dim symbolic; the fixed automaton widths ride as attrs,
    swallowed by ``_attrs``.
    """
    dtype = z_row.dtype
    r_dim, i_dim = z_row.shape[0], z_index.shape[0]
    p_dim, n_dim = prefix_sum.shape[0], next_prefix_sum.shape[0]
    num_vars = jnp.maximum(r_dim, p_dim)
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
        t_layer = frx.vmap(lambda tm: eq16 @ tm)(t_res)  # [4,4]
        return t_layer @ sv

    sv0 = (
        jnp.zeros(NUM_MEMORY_STATES, dtype=dtype)
        .at[_SUCCESS_INDEX]
        .set(jnp.ones([], dtype=dtype))
    )
    # num_vars layers (not num_vars+1): a layer == num_vars would read all-zero
    # bits (every dim <= num_vars) and is an identity on the carry-0 start
    # state, so it's a no-op — omit it.
    sv = frx.lax.fori_loop(0, num_vars, body, sv0)
    return sv[_INITIAL_INDEX]


def bp_eval_core(
    z_row: Array,
    z_index: Array,
    prefix_sum: Array,
    next_prefix_sum: Array,
    t_matrix: Array,
) -> Array:
    """h(z_row, z_index; t_c, t_{c+1}) — the per-column jagged BP indicator eval,
    wrapped in the `zorch.jagged_bp` name-routed composite so a vendor fuses the
    whole DP fold into one register-resident kernel (the CLAUDE.md fusion
    non-negotiable). The decomposition is byte-identical, so an unrecognized marker
    lowers with no behavior change. A ``vmap`` over columns (`_bp_all`) batches
    this single-column composite, mirroring Poseidon2's single-state permute
    marker.

    The layer count is NOT an operand: it is derived from the operand shapes
    (``max(z_row.shape[0], prefix_sum.shape[0])``) in the decomposition, since a
    bare scalar operand is constant-sunk into the fusion and would break the
    emitter's positional ABI. Deriving keeps a symbolic export dim symbolic.
    """
    return fused_region(
        _bp_eval_decomposition,
        z_row,
        z_index,
        prefix_sum,
        next_prefix_sum,
        t_matrix,
        name=JAGGED_BP_MARKER,
        version=JAGGED_BP_MARKER_VERSION,
        num_memory_states=NUM_MEMORY_STATES,
        num_bit_states=NUM_BIT_STATES,
    )
