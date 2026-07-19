# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Univariate polynomial evaluation, in evaluation and coefficient form."""

from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
from frx import Array

from zorch.utils.field import base_field, naturals


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
    out = fnp.ones((1,), dtype=x.dtype)
    step = x
    while out.shape[0] < n:
        out = fnp.concatenate([out, out * step])
        step = step * step
    return out[:n]


def eval_univariate(evals: Array, x: Array) -> Array:
    """Evaluate a univariate given by its values on ``[0, 1, ..., len-1]`` at
    ``x``, by Lagrange interpolation over that integer domain.

    A composer over the jitted basis kernel, so itself un-jitted."""
    nodes = naturals(evals.shape[0], evals.dtype)
    return fnp.dot(evals, compute_lagrange_basis(x, nodes))


def _lagrange_denominators(domain: Array) -> Array:
    """``prod_{j != k} (x_k - x_j)`` per node: exclude-self products via a
    masked matrix whose diagonal holds a typed one (a bare literal is not
    converted to a field dtype inside jit)."""
    n = domain.shape[0]
    one = fnp.ones((), domain.dtype)
    mask = fnp.eye(n, dtype=bool)
    return fnp.prod(fnp.where(mask, one, domain[:, None] - domain[None, :]), axis=1)


@frx.jit
def compute_lagrange_basis(r: Array, domain: Array) -> Array:
    """All Lagrange basis evaluations ``L_{D,k}(r)`` over ``domain``:
    ``L_{D,k}(r) = prod_{j != k} (r - x_j) / (x_k - x_j)``.

    Direct form, not barycentric — barycentric divides by ``(r - node)``,
    which an ``r`` landing on a node would zero."""
    one = fnp.ones((), r.dtype)
    mask = fnp.eye(domain.shape[0], dtype=bool)
    numerators = fnp.prod(fnp.where(mask, one, (r - domain)[None, :]), axis=1)
    return numerators / _lagrange_denominators(domain)


def compute_inv_vandermonde(degree: int, dtype: Any) -> Array:
    """Inverse Vandermonde over the natural domain ``{0..degree}``:
    ``coeffs = M @ evals`` for ``evals[j] = p(j)``.

    Built in the base field — the Lagrange basis for an integer domain lives
    in the prime field — so one matrix serves BF and EF callers; EF
    evaluations promote at multiply time."""
    base = base_field(dtype)
    n = degree + 1
    one = fnp.array(1, base)
    zero = fnp.array(0, base)
    domain = naturals(n, dtype)
    denoms = _lagrange_denominators(domain)
    # Column j = coefficients of L_j(x) = prod_{k != j} (x - k) / denom_j,
    # expanded by repeated (x - k) multiplication over the coefficient list.
    columns = []
    for j in range(n):
        num_coeffs = [one]
        for k in range(n):
            if k != j:
                neg_k = -fnp.array(k, base)
                expanded = [zero] * (len(num_coeffs) + 1)
                for i, c in enumerate(num_coeffs):
                    expanded[i] = expanded[i] + c * neg_k
                    expanded[i + 1] = expanded[i + 1] + c
                num_coeffs = expanded
        columns.append(fnp.stack(num_coeffs) / denoms[j])
    return fnp.stack(columns, axis=1)


@frx.jit
def eval_coeffs(coeffs: Array, point: Array) -> Array:
    """``p(point) = sum_i coeffs[..., i] * point**i`` — the coefficient-form
    dual of ``eval_univariate``.

    The power vector ``point**i`` is a ``lax.associative_scan`` (log-depth prefix
    product), then one batched dot with the coefficients. A *sequential* scan over
    the coefficient axis lowers to a ``while`` the GPU runtime launches host-side
    once per iteration, so an n-coefficient eval pays n launch latencies — WHIR's
    out-of-domain eval at ``n = 2^13`` measured ~550 ms of pure dispatch. The
    prefix product is O(log n) kernels instead, and each stage takes one array
    operand, so the degree never inflates the kernel's operand count (an unrolled
    power chain would, and overflow the GPU's 32 KB kernel-parameter space —
    ``ptxas: too much parameter space``). Field mul/add are exact and associative,
    so the tree-grouped prefix product and the batched sum are byte-identical to a
    sequential power-sum."""
    n = coeffs.shape[-1]
    # ``point**i`` as the prefix product of ``[1, point, point, …]`` (n entries),
    # laid out on the last axis so it dots the coefficients' degree axis directly.
    seq = fnp.where(
        fnp.arange(n) == 0,
        fnp.ones_like(point)[..., None],
        point[..., None],
    )
    powers = frx.lax.associative_scan(lambda a, b: a * b, seq, axis=-1)
    return fnp.sum(coeffs * powers, axis=-1)
