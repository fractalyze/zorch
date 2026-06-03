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
reduce/gather beyond it), so it stays wrappable by the single-kernel marker
without restructuring; `prove_composite` wraps a whole protocol in a
`zorch.sumcheck` composite that zkx fuses, reusing the round's own summand. The
verifier dual lives in `zorch.sumcheck.verifier`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.round import Round
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

# composite.name / composite.version of zorch's product-sumcheck marker; zkx's
# SumcheckStrategySelector keys recognition off this exact pair.
SUMCHECK_MARKER = "zorch.sumcheck"
SUMCHECK_MARKER_VERSION = 1


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


def fold_pair(p0: Array, p1: Array, r: Array) -> Array:
    """Fold one split pair at challenge `r`: P0 + r*(P1 - P0)."""
    return p0 + r * (p1 - p0)


def lift_to_domain(p0: Array, p1: Array, degree: int) -> Array:
    """Lift one split pair to the evaluation domain [0..degree]:
    f[u] = P0 + u*(P1 - P0), shape (degree+1, *P0.shape).

    The whole u-domain is built at once so the round poly stays one batched
    reduction (not degree+1 separate ones). `us` uses jnp.stack (not jnp.arange,
    whose iota is unsupported for extension dtypes) and is reshaped to broadcast
    over any leading batch dims of the factor."""
    us = jnp.stack([jnp.array(u, p0.dtype) for u in range(degree + 1)])
    return p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0)


def fold(state: Sequence[Array], r: Array) -> list[Array]:
    """Fold each MLE at challenge `r`: P0 + r*(P1 - P0). Halves width."""
    return [fold_pair(p0, p1, r) for (p0, p1) in split_halves(state)]


def factors_on_domain(state: Sequence[Array], degree: int) -> list[Array]:
    """Lift each split factor to the round's evaluation domain [0..degree]:
    f_k[u] = P0_k + u*(P1_k - P0_k), one array of shape (degree+1, *batch) each."""
    return [lift_to_domain(p0, p1, degree) for (p0, p1) in split_halves(state)]


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Product sumcheck: s = sum_x prod_k P_k(x), one factor per state entry."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def _combine(self, *factors: Array) -> Array:
        """Product summand `prod_k f_k`; the scan driver reads only this, so the
        round owns its summand and the driver stays summand-generic."""
        return reduce(operator.mul, factors)

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)), shape (degree+1, *batch).

        One batched reduction over the whole u-domain, so it lowers toward a
        single reduction kernel rather than degree+1 separate ones."""
        return jnp.sum(self._combine(*factors_on_domain(state, self.degree)), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, Array]:
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = fold(state, r[0])
        return state, transcript, msg


def prove_composite(
    round: SumcheckRound,
    factors: Sequence[Array],
    challenges: Array,
    *,
    eq_poly_index: int = -1,
    small_value: bool = False,
) -> Array:
    """Wrap a whole sumcheck in a `zorch.sumcheck` composite marker.

    A thin wrapper over `round`, not a second prover: the marker body runs the
    round's own `_round_poly` summand and the shared `fold` once per variable, so
    it stays correct for any summand when the marker inlines. zkx's
    `SumcheckStrategySelector` recognizes the marker and lowers the whole protocol
    to one fused `sumcheck_rounds:`/`sumcheck_svo:` kernel; unrecognized it inlines
    to the same round messages `prove` produces, flattened round-major to
    `[num_vars*(degree+1)]` -- the layout the fusion writes.

    Fiat-Shamir stays *outside* the marker: the `num_vars-1` already-sampled fold
    `challenges` are operands (so the region is pure round arithmetic), which is
    why this takes challenges rather than a transcript. Operand layout is the
    recognition contract `[factor tables][fold challenges]`; the
    `degree`/`num_vars`/`num_factors`/`eq_poly_index`/`small_value` attributes gate
    the compiler's baseline-vs-SVO choice -- zorch stays optimization-agnostic and
    just reports the structure. The selector fuses the product case where the
    factor-table count equals `degree`, eq (if any) carried as one of the factors.
    """
    if not factors:
        raise ValueError("prove_composite needs at least one factor table")
    if challenges.ndim != 1:
        raise ValueError(
            f"challenges must be a 1-D vector of fold scalars, got rank "
            f"{challenges.ndim}"
        )
    num_vars = log2_strict_usize(factors[0].shape[-1])
    if challenges.shape[0] != num_vars - 1:
        raise ValueError(
            f"need num_vars-1={num_vars - 1} fold challenges, "
            f"got {challenges.shape[0]}"
        )
    # Shared shape / even width are validated fail-loud by `split_halves` when the
    # marker body traces below -- not re-checked here, to keep one validator.

    def body(
        *operands: Array,
        degree: int,
        num_vars: int,
        num_factors: int,
        eq_poly_index: int,
        small_value: bool,
    ) -> Array:
        tables = list(operands[:num_factors])
        msgs = [round._round_poly(tables)]
        for r in operands[num_factors:]:  # one fold challenge per inter-round step
            tables = fold(tables, r)
            msgs.append(round._round_poly(tables))
        return jnp.concatenate(msgs)  # round-major [num_vars*(degree+1)]

    return lax.composite(body, name=SUMCHECK_MARKER, version=SUMCHECK_MARKER_VERSION)(
        *factors,
        *challenges,
        degree=round.degree,
        num_vars=num_vars,
        num_factors=len(factors),
        eq_poly_index=eq_poly_index,
        small_value=small_value,
    )
