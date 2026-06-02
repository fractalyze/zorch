# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multilinear polynomial ops over the boolean hypercube.

`eval_mle` evaluates an MLE (evals in lexicographic order) at a point via the
equality-polynomial inner product (`poly.eq.expand_eq_to_hypercube`); the
LSB-consecutive `mle_fold` binds one variable. Reusable pieces a PCS/IOP stands
on — Basefold is the first consumer.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import expand_eq_to_hypercube


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
