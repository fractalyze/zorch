# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared reference oracles for sumcheck tests."""

from __future__ import annotations

from collections.abc import Sequence

from jax import Array


def product(factors: Sequence[Array]) -> Array:
    """Reference product summand: prod_k factor_k (the product sumcheck claim)."""
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def eval_mle_oracle(evals: Array, point: Array) -> Array:
    """Reference multilinear eval: fold a (2^n,) MLE at `point` (MSB-first).

    A naive bind-one-variable-at-a-time oracle, deliberately independent of the
    production `zorch.poly.multilinear.eval_mle` so tests cross-check against a
    different implementation rather than confirm a shared one."""
    cur = evals
    for r in point:
        half = cur.shape[-1] // 2
        cur = cur[..., :half] + r * (cur[..., half:] - cur[..., :half])
    return cur[0]
