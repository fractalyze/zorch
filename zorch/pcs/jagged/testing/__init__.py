# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared jagged test fixtures — reference oracles."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.pcs.jagged.poly import JaggedStaticConfig, _decode_prefix_sums
from zorch.poly.eq import expand_eq_to_hypercube


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
