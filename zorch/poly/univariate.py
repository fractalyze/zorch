# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Univariate polynomial evaluation, in evaluation and coefficient form."""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array


def eval_univariate(evals: Array, x: Array) -> Array:
    """Evaluate a univariate given by its values on ``[0, 1, ..., len-1]`` at
    ``x``, by Lagrange interpolation over that integer domain.

    A composer over the jitted basis kernel, so itself un-jitted. Nodes are
    built per element — an iota over an extension dtype is unsupported."""
    nodes = jnp.stack([jnp.array(i, evals.dtype) for i in range(evals.shape[0])])
    return dot_unrolled(evals, compute_lagrange_basis(x, nodes))


def dot_unrolled(a: Array, v: Array) -> Array:
    """``a @ v`` contracting ``a``'s last axis, unrolled over ``v``'s static
    length as straight-line field ops.

    Not ``dot`` and not ``reduce(multiply)``: a field ``dot`` fused with an
    elementwise consumer miscompiles on CPU under ``jit``
    (fractalyze/jax#168). Meant for round-poly-sized contractions — the
    unroll is ``len(v)`` ops; exact field arithmetic makes the spelling
    byte-neutral."""
    return reduce(operator.add, (a[..., k] * v[k] for k in range(v.shape[0])))


def _lagrange_denominators(domain: Array) -> Array:
    """``prod_{j != k} (x_k - x_j)`` per node: exclude-self products via a
    masked matrix whose diagonal holds a typed one (a bare literal is not
    converted to a field dtype inside jit)."""
    n = domain.shape[0]
    one = jnp.ones((), domain.dtype)
    mask = jnp.eye(n, dtype=bool)
    return jnp.prod(jnp.where(mask, one, domain[:, None] - domain[None, :]), axis=1)


@jax.jit
def compute_lagrange_basis(r: Array, domain: Array) -> Array:
    """All Lagrange basis evaluations ``L_{D,k}(r)`` over ``domain``:
    ``L_{D,k}(r) = prod_{j != k} (r - x_j) / (x_k - x_j)``.

    Direct form, not barycentric — barycentric divides by ``(r - node)``,
    which an ``r`` landing on a node would zero."""
    one = jnp.ones((), r.dtype)
    mask = jnp.eye(domain.shape[0], dtype=bool)
    numerators = jnp.prod(jnp.where(mask, one, (r - domain)[None, :]), axis=1)
    return numerators / _lagrange_denominators(domain)


def compute_inv_vandermonde(degree: int, dtype: Any) -> Array:
    """Inverse Vandermonde over the natural domain ``{0..degree}``:
    ``coeffs = M @ evals`` for ``evals[j] = p(j)``.

    Built in the base field — the Lagrange basis for an integer domain lives
    in the prime field — so one matrix serves BF and EF callers; EF
    evaluations promote at multiply time."""
    try:
        base = zk_dtypes.efinfo(dtype).base_field_dtype
    except ValueError:
        base = dtype
    n = degree + 1
    one = jnp.array(1, base)
    zero = jnp.array(0, base)
    domain = jnp.array(list(range(n)), base)
    denoms = _lagrange_denominators(domain)
    # Column j = coefficients of L_j(x) = prod_{k != j} (x - k) / denom_j,
    # expanded by repeated (x - k) multiplication over the coefficient list.
    columns = []
    for j in range(n):
        num_coeffs = [one]
        for k in range(n):
            if k != j:
                neg_k = -jnp.array(k, base)
                expanded = [zero] * (len(num_coeffs) + 1)
                for i, c in enumerate(num_coeffs):
                    expanded[i] = expanded[i] + c * neg_k
                    expanded[i + 1] = expanded[i + 1] + c
                num_coeffs = expanded
        columns.append(jnp.stack(num_coeffs) / denoms[j])
    return jnp.stack(columns, axis=1)


@jax.jit
def eval_coeffs(coeffs: Array, point: Array) -> Array:
    """``p(point) = sum_i coeffs[..., i] * point**i`` — the coefficient-form
    dual of ``eval_univariate``.

    Horner over the static coefficient count: straight-line multiply-adds
    that broadcast over batched coefficient rows. Not a ``dot`` — a field
    ``dot`` fused with an elementwise consumer miscompiles on CPU under
    ``jit`` (fractalyze/jax#168), and exact field arithmetic makes the
    evaluation order byte-neutral. The unroll is coefficient-count-sized —
    round-poly inputs, not whole codewords."""
    result = coeffs[..., -1]
    for i in range(coeffs.shape[-1] - 2, -1, -1):
        result = result * point + coeffs[..., i]
    return result
