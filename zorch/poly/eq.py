# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Equality polynomial eq(w, x) hypercube expansion.

eq(w, x) = Π_i (1 - x_i - w_i + 2·x_i·w_i); Σ_{w∈{0,1}^n} eq(w,x) = 1.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def eq_factor(t: Array, z: Array) -> Array:
    """One coordinate's eq factor eq(t, z) = t·z + (1-t)(1-z), elementwise over
    any broadcastable shapes. Symmetric in ``t``/``z``.

    The building block ``eval_eq`` multiplies across coordinates, and the
    per-round factor a sumcheck binds one variable at a time: a prover that
    folds variable z_k at challenge r multiplies its running bound-eq mass by
    ``eq_factor(r, z_k)`` (the jagged provers' ``pad_adj``/``eq_adj``
    accumulation step)."""
    one = jnp.ones((), z.dtype)
    return t * z + (one - t) * (one - z)


def eq_root(z: Array) -> Array:
    """The root (in t) of the eq factor: ``eq_factor(eq_root(z), z) == 0`` at
    ``b = (1-z)/(1-2z)``, elementwise over any shape.

    Both sides derive ``b`` from ``z`` alone, so a round polynomial whose
    summand carries the current variable's eq factor has a known zero there --
    the free interpolation point of the Gruen round-poly compression
    (https://eprint.iacr.org/2024/108, `zorch.sumcheck.gruen`). Undefined at
    z = 1/2 (the factor is constant) and colliding with the t = 1 node at
    z = 0; a transcript-sampled ``z`` avoids both w.h.p."""
    one = jnp.ones((), z.dtype)
    return (one - z) / (one - jnp.array(2, z.dtype) * z)


def eval_eq(w: Array, x: Array) -> Array:
    """eq(w, x) = Π_i (1 - w_i - x_i + 2·w_i·x_i) for two equal-length points, in
    O(len) time and O(1) memory.

    The closed form of the equality polynomial evaluated at a pair of points --
    equals ``(expand_eq_to_hypercube(x, 1) · expand_eq_to_hypercube(w, 1)).sum()``
    but without materializing either 2^len vector, so a verifier evaluating eq at
    a bound point stays succinct. Symmetric in ``w``/``x`` and order-agnostic (a
    product over coordinates), so MSB/LSB indexing does not matter."""
    return jnp.prod(eq_factor(w, x), axis=-1)


def expand_hypercube_step(state: Array, coord: Array) -> Array:
    """(2^k,) -> (2^{k+1},): add a new variable as the LSB. result[2j] =
    state[j]·(1-coord), result[2j+1] = state[j]·coord."""
    high = state * coord
    low = state - high
    return jnp.column_stack([low, high]).flatten()


def contract_hypercube_step(state: Array) -> Array:
    """(2^{k+1},) -> (2^k,): Σ-marginalize the LSB variable by summing adjacent
    pairs, ``out[j] = state[2j] + state[2j+1]`` (over the last axis).

    The mass-preserving dual of `expand_hypercube_step`: expand splits each
    entry into ``(1-coord)``/``coord`` shares, so contracting recovers the
    pre-expansion table exactly -- ``Σ_b eq((w, b), x) = eq(w, x[:-1])``. A
    fixed-shape round loop that binds variables LSB-first keeps its eq table
    current with one of these per round instead of re-expanding. The last axis
    must be even."""
    return state[..., 0::2] + state[..., 1::2]


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
