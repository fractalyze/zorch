# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover rounds.

A sumcheck round splits each MLE on the current variable, sends the round
polynomial over the domain [0..degree], then folds every MLE at the verifier's
challenge (P0 + r*(P1 - P0)). The split/validate and fold steps are summand-
independent, so they live as `split_halves` / `fold` and each round supplies
only its summand via `_round_poly`: `SumcheckRound` (here) sums a product of
factors; `LogupSumcheckRound` (in zorch.logup_gkr.prover) sums the LogUp combine.

The round body is element-wise field ops plus the one inherent Sigma (no
reduce/gather beyond it), so it stays wrappable by the Phase-3 single-kernel
marker without restructuring (no fusion decorator now). The verifier dual lives
in `zorch.sumcheck.verifier`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import reduce

import jax.numpy as jnp
from jax import Array

from zorch.round import Round
from zorch.transcript import Transcript


def split_halves(state: Sequence[Array]) -> list[tuple[Array, Array]]:
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


def fold(state: Sequence[Array], r: Array) -> list[Array]:
    """Fold each MLE at challenge `r`: P0 + r*(P1 - P0). Halves width."""
    return [p0 + r * (p1 - p0) for (p0, p1) in split_halves(state)]


class SumcheckRound(Round):
    """Product sumcheck: s = sum_x prod_k P_k(x), one factor per state entry."""

    def __init__(self, degree: int) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)), shape (degree+1, *batch).

        The whole u-domain is evaluated at once -- one batched reduction, so it
        lowers toward a single reduction kernel rather than degree+1 separate
        ones. `us` is reshaped to broadcast over any leading batch dims of the
        factors, and is built with jnp.stack (not jnp.arange, whose iota is
        unsupported for extension dtypes)."""
        halves = split_halves(state)
        dtype = state[0].dtype
        us = jnp.stack([jnp.array(u, dtype) for u in range(self.degree + 1)])
        factors = (
            p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0) for (p0, p1) in halves
        )
        return jnp.sum(reduce(operator.mul, factors), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, Array]:
        msg = self._round_poly(state)
        transcript = transcript.observe(msg)
        transcript, r = transcript.sample(1)
        state = fold(state, r[0])
        return state, transcript, msg
