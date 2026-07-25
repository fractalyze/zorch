# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""EqPoly small-value sumcheck (Algorithm 6): the eq-weighted product sumcheck of
Algorithm 5, sped up over its first l₀ rounds by precomputed accumulators.

Three phases proving one sumcheck, so the messages match prove_eq_poly on the
same challenges (testing/small_value_test.py):

- Rounds 1..l₀ (SmallValueRound): the round polynomial is a contraction of the
  running R tensor against the round's accumulator; the factors are never touched,
  only R grows (by a Lagrange tensor factor) and the eq mass advances.
- Round l₀+1 (transition): materialize the postponed folds — one refold of
  [P₁, …, P_d, eq(w,·)] over the l₀ bound variables — then a TransitionRound over
  those factors, handing the tail its folded factors.
- Rounds l₀+2..l: the ordinary EqPolyRound tail.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.poly.eq import eq_factor, expand_eq_to_hypercube, expand_hypercube_step
from zorch.poly.univariate import compute_lagrange_basis
from zorch.prove import fold_rounds
from zorch.round import ProverRound
from zorch.sumcheck.domain import fold, summand_evals, uhat_domain
from zorch.sumcheck.eq.accumulators import precompute_accumulators
from zorch.sumcheck.eq.eq_poly import EqPolyRound, sumcheck_poly_from_t
from zorch.sumcheck.prover import ProductSummand
from zorch.sumcheck.sqrt_space import compute_folded_evaluations
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.utils.field import naturals

# (R_tensor, eq_w_prev, eq_evals): the contracted R tensor over U_dⁱ⁻¹, the running
# eq mass eq(w[<i], r[<i]), and the eq(r[<i], ·) table over the bound prefix — all
# length-1 at the start. eq_evals doubles each round, so the round index reads off it.
SmallValueState = tuple[Array, Array, Array]


def _lagrange_over_round_domain(r: Array, d: int) -> Array:
    """Lagrange basis over U_d at r: [L_∞, L₀, …, L_{d−1}], shape (d+1,). L_∞ is the
    vanishing polynomial on the finite nodes (the leading-coeff basis). Each round's
    R tensor grows by this factor."""
    finite = naturals(d, r.dtype)
    return fnp.concatenate(
        [fnp.atleast_1d(fnp.prod(r - finite)), compute_lagrange_basis(r, finite)]
    )


class SmallValueRound(ProverRound):
    """The accumulator round: sᵢ = lᵢ · (Rᵢ · Aᵢ), then grow R by the round's Lagrange
    tensor and advance the eq mass + table. One object drives all l₀ rounds under
    fold_rounds — it reads the round index off the eq table (which doubles each round)
    and picks that round's accumulator, the same shape as EqPolyRound.

    Product-bound: the accumulators were precomputed over Û_d against a product
    contraction (Procedure 9), so unlike EqPolyRound it takes no general summand."""

    def __init__(self, d: int, w: Array, accumulators: list[Array]) -> None:
        self.d = d
        self.w = w
        self.accumulators = accumulators
        self.domain = uhat_domain(d, w.dtype)

    def _round_poly(self, state: SmallValueState) -> Array:
        r_tensor, eq_w_prev, eq_evals = state
        i = log2_strict_usize(eq_evals.shape[0])  # 0-based round index
        t_evals = (r_tensor[:, None] * self.accumulators[i]).sum(axis=0)
        l_evals = expand_hypercube_step(eq_w_prev, self.w[i])
        return sumcheck_poly_from_t(t_evals, l_evals, self.domain)

    def __call__(
        self, state: SmallValueState, transcript: Transcript
    ) -> tuple[SmallValueState, Transcript, Array]:
        r_tensor, eq_w_prev, eq_evals = state
        i = log2_strict_usize(eq_evals.shape[0])
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        new_r = (
            r_tensor[:, None] * _lagrange_over_round_domain(r[0], self.d)[None, :]
        ).reshape(-1)
        new_state = (
            new_r,
            eq_w_prev * eq_factor(r[0], self.w[i]),
            expand_hypercube_step(eq_evals, r[0]),
        )
        return new_state, transcript, msg


class TransitionRound(ProverRound):
    """The √-space→eq-poly handoff (round l₀+1): one product round over the d+1 factors
    [P₁..P_d, eq(w,·)] at Û_d, binding the variable and advancing the eq mass for the
    tail. Runs on the materialized factors — the boundary compute_folded_evaluations
    has already collapsed the folds the accumulator phase postponed."""

    def __init__(self, d: int, w_l0: Array, dtype: Any) -> None:
        self.summand = ProductSummand(degree=d)
        self.w_l0 = w_l0
        self.domain = uhat_domain(d, dtype)

    def __call__(
        self, state: tuple[Array, Array], transcript: Transcript
    ) -> tuple[tuple[Array, Array], Transcript, Array]:
        folded, eq_w_prev = state
        msg = summand_evals(folded, self.summand._combine, self.domain)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return (
            (fold(folded, r[0]), eq_w_prev * eq_factor(r[0], self.w_l0)),
            transcript,
            msg,
        )


def _precompute(p_initial: Array, w: Array, l_0: int) -> tuple[list[Array], Array]:
    """Accumulators A_1..A_{l₀} and the factors with eq(w,·) appended as the
    (d+1)-th factor (the transition round folds all d+1 together)."""
    l = log2_strict_usize(p_initial.shape[1])
    l_half = l // 2
    one = fnp.ones((), dtype=p_initial.dtype)

    e_in = expand_eq_to_hypercube(w[l_0 : l_0 + l_half], one)
    w_x_out = w[l_half + l_0 :]
    x_out_size = 1 << (l - l_half - l_0)
    e_out = [
        fnp.reshape(
            expand_eq_to_hypercube(fnp.concatenate([w[i:l_0], w_x_out]), one),
            (1 << (l_0 - i), x_out_size),
        )
        for i in range(1, l_0 + 1)
    ]
    accumulators = precompute_accumulators(p_initial, e_in, e_out)
    p_with_weights = fnp.vstack([p_initial, expand_eq_to_hypercube(w, one)[None, :]])
    return accumulators, p_with_weights


def prove_eq_poly_small_value(
    p_initial: Array, w: Array, l_0: int, transcript: Transcript
) -> tuple[Array, Transcript, list[Array]]:
    """Prove the eq-weighted sumcheck with l₀ small-value rounds. Returns the final
    folded factors (d, 1), the transcript, and all l round messages (each over Û_d)."""
    d = p_initial.shape[0]
    l = log2_strict_usize(p_initial.shape[1])
    if not 1 <= l_0 <= l - l // 2:
        raise ValueError(
            f"l_0 must be in [1, {l - l // 2}] (the small-value rounds fit before "
            f"the out-half); got l_0={l_0} for l={l}"
        )
    accumulators, p_with_weights = _precompute(p_initial, w, l_0)

    # Phase 1: the accumulator rounds under the standard host-loop driver. The state
    # carries the eq table the transition needs, so nothing is captured by hand.
    one = fnp.ones(1, dtype=p_initial.dtype)
    state, transcript, msgs = fold_rounds(
        SmallValueRound(d, w, accumulators), (one, one, one), transcript, l_0
    )
    _, eq_w_prev, eq_evals = state

    # Phase 2: the √-space→dense boundary. Materialize the postponed folds (as in
    # sqrt_space), then one transition round over the d+1 factors [P₁..P_d, eq(w,·)].
    # Sampling at Û_d (not Û_{d+1}) drops the eq factor from the message, its fold
    # riding the scalar eq mass, so the tail keeps only the d real factors.
    folded = compute_folded_evaluations(p_with_weights, eq_evals)
    (folded, eq_w_prev), transcript, msg_t = TransitionRound(
        d, w[l_0], p_initial.dtype
    )((folded, eq_w_prev), transcript)
    folded_p = folded[:d]

    # Phase 3: the ordinary eq-poly tail. Product-bound: the accumulator precompute
    # (Procedure 9) contracts a product, so this engine is a product sumcheck only —
    # unlike EqPolyRound / SqrtSpaceRound, it does not take a general summand.
    (p_final, _), transcript, tail = fold_rounds(
        EqPolyRound(ProductSummand(degree=d), w),
        (folded_p, eq_w_prev),
        transcript,
        l - l_0 - 1,
    )
    return p_final, transcript, msgs + [msg_t] + tail
