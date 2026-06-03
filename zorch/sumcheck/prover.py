# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover rounds.

A sumcheck round splits each MLE on the current variable, sends the round
polynomial over the domain [0..degree], then folds every MLE at the verifier's
challenge (P0 + r*(P1 - P0)). The split/validate and fold steps are summand-
independent, so they live as `split_halves` / `factors_on_domain` / `fold` and
each round supplies only its summand via `_round_poly`: `SumcheckRound` (here)
sums a product of factors; `LogupSumcheckRound` (in zorch.logup_gkr.prover) sums
the LogUp combine.

The round body is element-wise field ops plus the one inherent Sigma (no
reduce/gather beyond it), so it stays wrappable by the Phase-3 single-kernel
marker without restructuring (no fusion decorator now). The verifier dual lives
in `zorch.sumcheck.verifier`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce

import jax
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


def factors_on_domain(state: Sequence[Array], degree: int) -> list[Array]:
    """Lift each split factor to the round's evaluation domain [0..degree]:
    f_k[u] = P0_k + u*(P1_k - P0_k), one array of shape (degree+1, *batch) each.

    The whole u-domain is built at once so the round poly stays one batched
    reduction (not degree+1 separate ones). `us` uses jnp.stack (not jnp.arange,
    whose iota is unsupported for extension dtypes) and is reshaped to broadcast
    over any leading batch dims of the factors."""
    halves = split_halves(state)
    dtype = state[0].dtype
    us = jnp.stack([jnp.array(u, dtype) for u in range(degree + 1)])
    return [p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0) for (p0, p1) in halves]


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Product sumcheck: s = sum_x prod_k P_k(x), one factor per state entry."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)), shape (degree+1, *batch).

        One batched reduction over the whole u-domain, so it lowers toward a
        single reduction kernel rather than degree+1 separate ones."""
        return jnp.sum(
            reduce(operator.mul, factors_on_domain(state, self.degree)), axis=-1
        )

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, Array]:
        msg = self._round_poly(state)
        transcript = transcript.observe(msg)
        transcript, r = transcript.sample(1)
        state = fold(state, r[0])
        return state, transcript, msg
