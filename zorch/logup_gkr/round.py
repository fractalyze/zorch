# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR as a `Round`: one per-variable round of a GKR layer.

A `SumcheckRoundBase` whose summand is the LogUp combine

    eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)

over five MLE factors in order `[eq, n0, d1, n1, d0]`. The round polynomial is
degree 3 (eq is degree 1, the bracket degree 2). The fold, the Fiat-Shamir loop,
the halving, and the shape/even-width checks come from `SumcheckRoundBase`; this
class adds only the combine and the five-factor count check.

This is the per-variable round only. The full LogUp-GKR protocol -- fractional-sum
circuit, cross-layer GKR transitions, jagged/interaction layout, verifier -- is a
separate, later piece.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from zorch.sumcheck.prover import SumcheckRoundBase

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2).
_DEGREE = 3
_NUM_FACTORS = 5  # [eq, n0, d1, n1, d0]


class LogupGkrRound(SumcheckRoundBase):
    def __init__(self, lam: Array) -> None:
        # Batching challenge; fixed across a layer's variable-rounds.
        self.lam = lam

    def _split(self, state: Sequence[Array]) -> list[tuple[Array, Array]]:
        # Factor count is LogUp-specific; shape/even-width checks come from base.
        if len(state) != _NUM_FACTORS:
            raise ValueError(
                f"state must hold {_NUM_FACTORS} factors [eq, n0, d1, n1, d0], "
                f"got {len(state)}"
            )
        return super()._split(state)

    def _combine(self, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array) -> Array:
        return eq * (self.lam * (n0 * d1 + n1 * d0) + d0 * d1)

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """Round polynomial over the domain [0, 1, ..., degree], shape
        (degree+1, *batch): s[u] = sum_x' combine(f_u for each factor), where
        f_u = P0 + u*(P1 - P0).

        The whole u-domain is evaluated at once -- one batched reduction. `us` is
        reshaped to broadcast over any leading batch dims of the factors, and is
        built with jnp.stack (not jnp.arange, whose iota is unsupported for
        extension dtypes)."""
        halves = self._split(state)
        dtype = state[0].dtype
        us = jnp.stack([jnp.array(u, dtype) for u in range(_DEGREE + 1)])
        factors = [
            p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0) for (p0, p1) in halves
        ]
        return jnp.sum(self._combine(*factors), axis=-1)
