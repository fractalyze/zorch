# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared jagged test fixtures — reference oracles.

``JaggedStaticConfig`` + ``eval_jagged_mle`` live here, not in production
``poly.py``: the production cores are shape-polymorphic (every dim derived from
array shapes), so they need no static config. This static-config bundle is purely
oracle convenience — the differential references (``scatter_partial_eval``,
``eval_jagged_mle``) and the tests that cross-check the shape-polymorphic
``partial_eval_core`` against them.
"""

from dataclasses import dataclass
from functools import partial
from functools import reduce as _reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.pcs.jagged.branching_program import _TRANSITION_ROWS, bp_eval_core
from zorch.pcs.jagged.poly import (
    _decode_prefix_sums,
    build_jagged_layout,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class JaggedStaticConfig:
    """Static dim bundle for the oracles (frozen -> hashable for jit
    static_argnames). l_max: column count.  n_c = ceil(log2 l_max).  n_r: row-bit
    width.  n_d: BP layer count (log-area tier).  dtype: field dtype."""

    l_max: int
    n_c: int
    n_r: int
    n_d: int
    dtype: object


def oracle_cfg(
    col_heights: list[int], l_max: int, n_r: int, dtype: Any
) -> JaggedStaticConfig:
    """Build the oracle's static config (n_d via ``build_jagged_layout``)."""
    _, n_d = build_jagged_layout(col_heights, l_max, dtype)
    return JaggedStaticConfig(
        l_max=l_max, n_c=log2_ceil_usize(l_max), n_r=n_r, n_d=n_d, dtype=dtype
    )


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

    The branching-program reference for ``partial_eval_core``: ``col_prefix_sums``
    is ``build_jagged_layout``'s FIELD bit tensor.  Output shape () (a function of
    shapes). jnp.sum aborts on EF, so the column sum is a trace-time
    functools.reduce over the static ``cfg.l_max`` (fine for the verifier-once /
    small-l_max oracle)."""
    dtype = z_row.dtype
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones([], dtype=dtype))  # 2^{n_c}
    all_left = col_prefix_sums[: cfg.l_max]
    all_right = col_prefix_sums[1:]
    bp_evals = jax.vmap(
        lambda pl, pr: bp_eval_core(z_row, z_index, pl, pr, t_matrix),
        in_axes=(0, 0),
    )(all_left, all_right)
    return _reduce(
        lambda a, b: a + b, [col_eq[c] * bp_evals[c] for c in range(cfg.l_max)]
    )


@partial(jax.jit, static_argnames=("cfg",))
def scatter_partial_eval(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    *,
    cfg: JaggedStaticConfig,
) -> Array:
    """Per-column scatter form of `partial_eval` — the bit-exactness oracle.

    Column c adds col_eq[c]·row_eq masked to its height into
    out[t_c : t_c + 2^n_r] through a loop-carried RMW (serial on GPU); the
    production gather form must match it byte-for-byte, including the
    taller-than-capacity truncation at 2^n_r.
    """
    dtype = z_row.dtype
    row_len = 1 << cfg.n_r  # static

    prefix_sums_int = _decode_prefix_sums(col_prefix_sums, cfg.n_d)

    col_eq = expand_eq_to_hypercube(z_col, jnp.ones([], dtype=dtype))  # [2^n_c]
    row_eq = expand_eq_to_hypercube(z_row, jnp.ones([], dtype=dtype))  # [2^n_r]

    out = jnp.zeros(1 << cfg.n_d, dtype=dtype)
    row_indices = jnp.arange(row_len, dtype=jnp.int32)

    def body(c: Array, out: Array) -> Array:
        t_c = prefix_sums_int[c]
        h = prefix_sums_int[c + 1] - t_c  # column height (0 for padding columns)

        # Mask row entries beyond this column's actual height.
        mask = row_indices < h  # (2^n_r,) bool
        contrib = col_eq[c] * jnp.where(mask, row_eq, jnp.zeros([], dtype=dtype))

        # Read-modify-write: add contrib into out[t_c : t_c + row_len].
        old = jax.lax.dynamic_slice(out, (t_c,), (row_len,))
        return jax.lax.dynamic_update_slice(out, old + contrib, (t_c,))

    return jax.lax.fori_loop(0, cfg.l_max, body, out)
