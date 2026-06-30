# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Univariate polynomial evaluation, in evaluation and coefficient form."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array


def powers(x: Array, n: int) -> Array:
    """``(1, x, x², …, x^{n-1})`` ascending, length ``n`` (``n`` static).

    Built by log-doubling (``powers[m:2m] = powers[:m]·xᵐ``) so the traced graph
    is O(log n), not ``n`` unrolled multiplies — a linear power chain makes the
    fused kernel's operand count scale with ``n`` and past a few thousand entries
    overruns the GPU's kernel-parameter space (the same cliff ``eval_coeffs``
    avoids). The monomial-basis evaluation vector: ``⟨coeffs, powers(x, n)⟩ =
    Σ cᵢ xⁱ``."""
    if n < 1:
        raise ValueError(f"powers needs n >= 1, got {n}")
    out = jnp.ones((1,), dtype=x.dtype)
    step = x
    while out.shape[0] < n:
        out = jnp.concatenate([out, out * step])
        step = step * step
    return out[:n]


def eval_univariate(evals: Array, x: Array) -> Array:
    """Evaluate a univariate given by its values on ``[0, 1, ..., len-1]`` at
    ``x``, by Lagrange interpolation over that integer domain.

    A composer over the jitted basis kernel, so itself un-jitted. Nodes are
    built per element — an iota over an extension dtype is unsupported."""
    nodes = jnp.stack([jnp.array(i, evals.dtype) for i in range(evals.shape[0])])
    return jnp.dot(evals, compute_lagrange_basis(x, nodes))


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

    The power-sum is carried through a ``lax.scan`` over the coefficient axis
    (degree moved to the leading axis): the carry threads
    ``(accumulator, point**i)``, keeping the traced graph O(1) in the degree so
    the fused kernel takes one array operand regardless of ``n``. A
    coefficient-count-dependent graph (e.g. an explicit power chain) makes the
    fused kernel's operand count scale with ``n``, which past a few thousand
    coefficients exceeds the GPU's 32 KB kernel-parameter space and fails to
    compile (``ptxas: too much parameter space``) — WHIR's out-of-domain eval
    at large stacked sizes hits exactly this. Field arithmetic is exact, so the
    scan is byte-identical to the direct power-sum."""
    leading = jnp.moveaxis(coeffs, -1, 0)  # (n, *batch): degree on axis 0

    # Forward power accumulation rather than a reverse-scan Horner (which would
    # carry only the accumulator): ``lax.scan(reverse=True)`` is not honored on
    # this jax fork — it runs forward and yields the wrong value.
    def step(
        carry: tuple[Array, Array], c_i: Array
    ) -> tuple[tuple[Array, Array], None]:
        acc, power = carry
        return (acc + c_i * power, power * point), None

    # acc seeds in the coeff×point promoted dtype/shape; power seeds at point**0.
    init = (jnp.zeros_like(coeffs[..., 0] * point), jnp.ones_like(point))
    (acc, _), _ = jax.lax.scan(step, init, leading)
    return acc
