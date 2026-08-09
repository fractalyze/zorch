# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Equality polynomial eq(w, x) hypercube expansion.

eq(w, x) = Π_i (1 - x_i - w_i + 2·x_i·w_i); Σ_{w∈{0,1}^n} eq(w,x) = 1.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array


def eq_factor(t: Array, z: Array) -> Array:
    """One coordinate's eq factor eq(t, z) = t·z + (1-t)(1-z), elementwise over
    any broadcastable shapes. Symmetric in ``t``/``z``.

    The building block ``eval_eq`` multiplies across coordinates, and the
    per-round factor a sumcheck binds one variable at a time: a prover that
    folds variable z_k at challenge r multiplies its running bound-eq mass by
    ``eq_factor(r, z_k)`` (the jagged provers' ``pad_adj``/``eq_adj``
    accumulation step)."""
    one = fnp.ones((), z.dtype)
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
    one = fnp.ones((), z.dtype)
    return (one - z) / (one - fnp.array(2, z.dtype) * z)


def eval_eq(w: Array, x: Array) -> Array:
    """eq(w, x) = Π_i (1 - w_i - x_i + 2·w_i·x_i) for two equal-length points, in
    O(len) time and O(1) memory.

    The closed form of the equality polynomial evaluated at a pair of points --
    equals ``(expand_eq_to_hypercube(x, 1) · expand_eq_to_hypercube(w, 1)).sum()``
    but without materializing either 2^len vector, so a verifier evaluating eq at
    a bound point stays succinct. Symmetric in ``w``/``x`` and order-agnostic (a
    product over coordinates), so MSB/LSB indexing does not matter."""
    return fnp.prod(eq_factor(w, x), axis=-1)


def expand_hypercube_step(state: Array, coord: Array, *, msb: bool = False) -> Array:
    """(2ᵏ,) -> (2ᵏ⁺¹,): add a new variable's (1-coord)/coord split — LSB (default)
    interleaves the shares, msb=True concatenates [low, high]."""
    high = state * coord
    low = state - high
    if msb:
        return fnp.concatenate([low, high])
    return fnp.column_stack([low, high]).flatten()


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


# Above this variable count, the table is built as an outer product of two
# half tables instead of one doubling chain. XLA fuses a chain's tail layers
# into a single concatenate fusion that recomputes each output element's
# ancestor factors (one multiply per fused layer per element), so a long chain
# pays several multiplies per entry and re-materializes every doubling; the
# outer product is exactly one multiply per entry over two small inputs.
_OUTER_SPLIT_MIN = 16


def expand_eq_to_hypercube(x: Array, scalar: Array, *, msb: bool = False) -> Array:
    """scalar·eq(w, x) for all w in {0,1}^n. `msb=False` interleaves each new share
    (default, `w[0]` the MSB); `msb=True` concatenates `[low, high]`, placing
    `x[j]` at bit `j`.

    NOTE: explicit indexing instead of `for coord in x` — iterating a JAX array
    of an extension-field dtype dispatches `lax.sign`, a XLA gotcha.
    """
    n = x.shape[0]
    if n >= _OUTER_SPLIT_MIN:
        # out[w] factors over any coordinate split, so the full table is the
        # outer product of the two half tables — with the slow-axis half being
        # whichever slice owns the high index bits (x[:k] when w[0] is the MSB,
        # x[k:] when msb=True places x[j] at bit j). GF multiplication is
        # exact, so the product is byte-equal to the chain.
        k = n // 2
        first = expand_eq_to_hypercube(x[:k], scalar, msb=msb)
        rest = expand_eq_to_hypercube(x[k:], fnp.ones((), x.dtype), msb=msb)
        outer, inner = (rest, first) if msb else (first, rest)
        return (outer[:, None] * inner[None, :]).reshape(-1)
    state = fnp.atleast_1d(scalar)
    for j in range(n):
        state = expand_hypercube_step(state, x[j], msb=msb)
    return state


def expand_monomial_step(state: Array, coord: Array) -> Array:
    """(2^k,) -> (2^{k+1},): add a new variable as the LSB, monomial basis.
    result[2j] = state[j], result[2j+1] = state[j]·coord — the ⊗(1, coord)
    factor, where `expand_hypercube_step` is ⊗(1-coord, coord)."""
    return fnp.column_stack([state, state * coord]).flatten()


def expand_monomial_to_hypercube(x: Array, scalar: Array) -> Array:
    """scalar·Π_{i: w_i=1} x_i for all w in {0,1}^n — the monomial
    (coefficient-basis) dual of `expand_eq_to_hypercube`, same (2^n,) shape and
    MSB-first indexing (w[0] binds x[0]). `<coeffs, expand_monomial(x)>` is the
    monomial-basis evaluation at x, as `<evals, expand_eq(x)>` is the eval-basis
    one."""
    state = fnp.atleast_1d(scalar)
    for j in range(x.shape[0]):
        state = expand_monomial_step(state, x[j])
    return state
