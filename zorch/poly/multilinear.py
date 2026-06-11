# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multilinear polynomial ops over the boolean hypercube.

`eval_mle` evaluates an MLE (evals in lexicographic order) at a point via the
equality-polynomial inner product (`poly.eq.expand_eq_to_hypercube`); the
LSB-consecutive `mle_fold` binds one variable; `mle_coeffs_to_evals` /
`mle_evals_to_coeffs` convert between the monomial-coefficient and
hypercube-evaluation bases. Reusable pieces a PCS/IOP stands on — Basefold is
the first consumer.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.utils.bits import log2_strict_usize


def _butterfly_scan(table: Array, combine: Callable[[Array, Array], Array]) -> Array:
    """Run the `k = log2` per-bit butterfly passes of a zeta/Möbius transform as
    one fixed-shape `lax.scan`, NOT a `k`-deep Python unroll.

    A naive unroll reshapes to a bit-`b`-dependent shape each pass, so it lowers
    to `k` distinct kernels the compiler can't recognise — the hand-rolled
    butterfly `coding.reed_solomon` warns against. Instead each level acts on the
    *current MSB* (a fixed `(2, n/2)` reshape) and then left-rotates the bit
    labels (`swapaxes(-2, -1)`), so the body is shape-invariant and scans; after
    `k` rotations the bit order is restored. The passes touch disjoint bits and
    field add/sub is commutative, so processing MSB-first scans is byte-identical
    to the LSB-first unroll. `combine(lo, hi) -> new_hi` is `lo + hi` (coeffs→
    evals) or `hi - lo` (evals→coeffs). Leading axes ride through."""
    lead = table.shape[:-1]
    n = table.shape[-1]
    k = log2_strict_usize(n)
    if k == 0:
        return table  # zero variables: the transform is the identity

    def level(a: Array, _: None) -> tuple[Array, None]:
        x = a.reshape(lead + (2, n // 2))
        lo, hi = x[..., 0, :], x[..., 1, :]
        paired = jnp.stack([lo, combine(lo, hi)], axis=-2)
        # Left-rotate the bit labels: the MSB just folded drops to the LSB, so
        # the next level's fixed (2, n/2) reshape exposes the next bit.
        return paired.swapaxes(-2, -1).reshape(lead + (n,)), None

    out, _ = lax.scan(level, table, xs=None, length=k)
    return out


def mle_coeffs_to_evals(coeffs: Array) -> Array:
    """Multilinear coefficient→evaluation (the zeta/subset-sum transform) over
    the trailing axis of `(..., 2ᵏ)`: the hypercube evaluation at vertex `v` is
    the sum of every coefficient whose monomial support is a subset of `v`. Runs
    the per-bit passes (`a[v] += a[v with one set bit cleared]`) as one fixed
    `lax.scan` (see `_butterfly_scan`). Inverse of `mle_evals_to_coeffs`; leading
    axes ride through."""
    return _butterfly_scan(coeffs, lambda lo, hi: lo + hi)


def mle_evals_to_coeffs(evals: Array) -> Array:
    """Evaluation→coefficient transform, the Möbius inverse of
    `mle_coeffs_to_evals` (`a[v] -= a[v with one set bit cleared]`), as one fixed
    `lax.scan`. Leading axes ride through."""
    return _butterfly_scan(evals, lambda lo, hi: hi - lo)


def eval_mle(mle: Array, point: Array, axis: int = 0) -> Array:
    """Evaluate an MLE at `point` via the eq inner product. Contracts `axis`
    (size 2ⁿ); leading/trailing axes ride through. 1-D MLE -> scalar."""
    eq = expand_eq_to_hypercube(point, jnp.ones((), mle.dtype))
    shape = [1] * mle.ndim
    shape[axis] = eq.shape[0]
    return (mle * eq.reshape(shape)).sum(axis=axis)


def mle_fold(evals: Array, beta: Array) -> Array:
    """Fold a consecutive-LSB variable pair: result[i] = evals[2i] + β·evals[2i+1].

    This is the additive Basefold/FRI combine (e0 + β·e1), NOT the multilinear
    partial-evaluation bind (1−β)·e0 + β·e1 that SumcheckRound uses. Acts on the
    last axis (`(..., 2ⁿ) -> (..., 2ⁿ⁻¹)`), so leading batch axes ride through.
    """
    pairs = evals.reshape(*evals.shape[:-1], -1, 2)
    return pairs[..., 0] + beta * pairs[..., 1]
