# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Equality polynomial eq(w, x) hypercube expansion.

eq(w, x) = Π_i (1 - x_i - w_i + 2·x_i·w_i); Σ_{w∈{0,1}^n} eq(w,x) = 1.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def expand_hypercube_step(state: Array, coord: Array) -> Array:
    """(2^k,) -> (2^{k+1},): add a new variable as the LSB. result[2j] =
    state[j]·(1-coord), result[2j+1] = state[j]·coord."""
    high = state * coord
    low = state - high
    return jnp.column_stack([low, high]).flatten()


def expand_eq_to_hypercube(x: Array, scalar: Array) -> Array:
    """scalar·eq(w, x) for all w in {0,1}^n. Returns (2^n,); result[nat(w)] with
    w[0] as the MSB.

    NOTE: explicit indexing instead of `for coord in x` — iterating a JAX array
    of an extension-field dtype dispatches `lax.sign`, a ZKX gotcha.
    """
    state = jnp.atleast_1d(scalar)
    for j in range(x.shape[0]):
        state = expand_hypercube_step(state, x[j])
    return state
