# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover rounds.

`SumcheckRoundBase` is the protocol skeleton -- split each MLE on the current
variable, send the round polynomial over the domain [0..degree], fold every MLE
at the verifier's challenge (P0 + r*(P1 - P0)). A subclass supplies only its
summand via `_round_poly`: `SumcheckRound` (here) sums a product of factors,
`LogupGkrRound` (in zorch.logup_gkr) sums the LogUp combine. The fold, the
Fiat-Shamir loop, and `_split` (which validates then halves) are inherited; a
subclass overrides `_split` only for extra input checks (e.g. LogUp's factor
count).

The round body is element-wise field ops plus the one inherent Sigma (no
reduce/gather beyond it), so it stays wrappable by the Phase-3 single-kernel
marker without restructuring (no fusion decorator now). The verifier dual lives
in `zorch.sumcheck.verifier`.
"""

from __future__ import annotations

import operator
from functools import reduce

import jax.numpy as jnp
from jax import Array

from zorch.round import Round


class SumcheckRoundBase(Round):
    """Generic sumcheck round; subclasses implement `_round_poly`."""

    def _split(self, state):
        """Validate, then halve each MLE on the current variable: [(P0, P1), ...].

        Factors must be non-empty, share a shape, and have an even width -- fail
        loud rather than silently drop the odd element on `// 2`."""
        if not state:
            raise ValueError("state must hold at least one factor")
        shape = state[0].shape
        out = []
        for i, evals in enumerate(state):
            if evals.shape != shape:
                raise ValueError(
                    f"all factors must share a shape; factor {i} is {evals.shape}, "
                    f"factor 0 is {shape}"
                )
            if evals.shape[-1] % 2 != 0:
                raise ValueError(f"factor width must be even, got {evals.shape[-1]}")
            half = evals.shape[-1] // 2
            out.append((evals[..., :half], evals[..., half:]))
        return out

    def _round_poly(self, state) -> Array:
        """Round polynomial over [0..degree] (degree set by the subclass's
        summand), shape (degree+1, *batch). Supplied by subclasses."""
        raise NotImplementedError

    def _fold(self, state, r) -> list:
        """Fold each MLE at challenge `r`: P0 + r*(P1 - P0). Halves width."""
        return [p0 + r * (p1 - p0) for (p0, p1) in self._split(state)]

    def __call__(self, state, transcript):
        msg = self._round_poly(state)
        transcript = transcript.observe(msg)
        transcript, r = transcript.sample(1)
        state = self._fold(state, r[0])
        return state, transcript, msg


class SumcheckRound(SumcheckRoundBase):
    """Product sumcheck: s = sum_x prod_k P_k(x), one factor per state entry."""

    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

    def _round_poly(self, state) -> Array:
        """s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)).

        The whole u-domain is evaluated at once -- one batched reduction, so it
        lowers toward a single reduction kernel rather than degree+1 separate
        ones. The u-domain is built with jnp.stack (not jnp.arange, whose iota
        is unsupported for extension dtypes)."""
        halves = self._split(state)
        dtype = state[0].dtype
        us = jnp.stack([jnp.array(u, dtype) for u in range(self.degree + 1)])
        factors = (p0 + us[:, None] * (p1 - p0) for (p0, p1) in halves)
        return jnp.sum(reduce(operator.mul, factors), axis=-1)
