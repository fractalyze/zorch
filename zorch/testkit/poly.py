# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only polynomial helpers shared across protocol tests."""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def eval_univariate(evals: Array, x) -> Array:
    """Evaluate a univariate given by its values on ``[0, 1, ..., len-1]`` at
    ``x``, via Lagrange interpolation over the field (small domain). Nodes are
    built per-element (an iota over an extension dtype is unsupported)."""
    d = evals.shape[0]
    nodes = [jnp.array(i, evals.dtype) for i in range(d)]
    acc = jnp.zeros((), evals.dtype)
    for i in range(d):
        num = jnp.ones((), evals.dtype)
        den = jnp.ones((), evals.dtype)
        for j in range(d):
            if j != i:
                num = num * (x - nodes[j])
                den = den * (nodes[i] - nodes[j])
        acc = acc + evals[i] * (num / den)
    return acc
