# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR as a `Round`: one per-variable round of a GKR layer.

State is `list[Array]` of five MLE evals in combine order `[eq, n0, d1, n1, d0]`;
the round sums the LogUp combine

    eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)

over the hypercube. The round polynomial is degree 3 (eq is degree 1, the
bracket degree 2). Like `SumcheckRound`, the body is element-wise field ops plus
the one inherent Sigma (no reduce/gather beyond it), so it stays wrappable by the
Phase-3 single-kernel marker without restructuring (no fusion decorator now).

This is the per-variable round only. The full LogUp-GKR protocol -- fractional-sum
circuit, cross-layer GKR transitions, jagged/interaction layout, verifier -- is a
separate, later piece.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.round import Round

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2).
_DEGREE = 3
_NUM_FACTORS = 5  # [eq, n0, d1, n1, d0]


class LogupGkrRound(Round):
    def __init__(self, lam):
        # Batching challenge; fixed across a layer's variable-rounds.
        self.lam = lam

    def _split(self, state):
        if len(state) != _NUM_FACTORS:
            raise ValueError(
                f"state must hold {_NUM_FACTORS} factors [eq, n0, d1, n1, d0], "
                f"got {len(state)}"
            )
        out = []
        for evals in state:
            half = evals.shape[-1] // 2
            out.append((evals[..., :half], evals[..., half:]))
        return out

    def _combine(self, eq, n0, d1, n1, d0):
        return eq * (self.lam * (n0 * d1 + n1 * d0) + d0 * d1)

    def round_poly(self, state) -> Array:
        """Round polynomial over the domain [0, 1, ..., degree], shape
        (degree+1,): s[u] = sum_x' combine(f_u for each factor), where
        f_u = P0 + u*(P1 - P0).

        The whole u-domain is evaluated at once -- one batched reduction. The
        u-domain is built with jnp.stack (not jnp.arange, whose iota is
        unsupported for extension dtypes)."""
        halves = self._split(state)
        dtype = state[0].dtype
        us = jnp.stack([jnp.array(u, dtype) for u in range(_DEGREE + 1)])
        factors = [p0 + us[:, None] * (p1 - p0) for (p0, p1) in halves]
        return jnp.sum(self._combine(*factors), axis=-1)

    def fold(self, state, r) -> list:
        """Fold each factor at challenge `r`: P0 + r*(P1 - P0). Halves width."""
        return [p0 + r * (p1 - p0) for (p0, p1) in self._split(state)]

    def __call__(self, state, transcript):
        msg = self.round_poly(state)
        transcript = self.commit(transcript, msg)
        transcript, r = self.challenge(transcript, 1)
        state = self.fold(state, r[0])
        return state, transcript, msg
