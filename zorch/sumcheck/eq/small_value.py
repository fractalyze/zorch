# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""EqPoly small-value sumcheck (Algorithm 6): the eq-weighted product sumcheck of
Algorithm 5, sped up over its first l₀ rounds by precomputed accumulators.

Three phases proving one sumcheck, so the messages match prove_eq_poly on the
same challenges (testing/small_value_test.py):

- Rounds 1..l₀ (SmallValueRound): the round polynomial is a contraction of the
  running R tensor against the round's accumulator; the factors are never touched,
  only R grows (by a Lagrange tensor factor) and the eq mass advances.
- Round l₀+1 (transition): one SqrtSpace-style refold of [P₁, …, P_d, eq(w,·)] over
  the l₀ bound variables, giving back a foldable factor state for the tail.
- Rounds l₀+2..l: the ordinary EqPolyRound tail.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import eq_factor, expand_eq_to_hypercube, expand_hypercube_step
from zorch.poly.univariate import compute_lagrange_basis
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import fold_stacked, product_round_poly
from zorch.sumcheck.eq.accumulators import precompute_accumulators
from zorch.sumcheck.eq.eq_poly import EqPolyRound, sumcheck_poly_from_t
from zorch.sumcheck.prover import SumcheckRound
from zorch.sumcheck.sqrt_space import compute_folded_evaluations
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

# (R_tensor, eq_w_prev): the contracted R tensor over U_dⁱ⁻¹ and the running eq
# mass eq(w[<i], r[<i]), both length-1 at the start.
SmallValueState = tuple[Array, Array]


def _lagrange_over_round_domain(r: Array, d: int) -> Array:
    """Lagrange basis over U_d at r: [L_∞, L₀, …, L_{d−1}], shape (d+1,). L_∞ is the
    vanishing polynomial on the finite nodes (the leading-coeff basis). Each round's
    R tensor grows by this factor."""
    finite = jnp.stack([jnp.array(k, r.dtype) for k in range(d)])
    return jnp.concatenate(
        [jnp.atleast_1d(jnp.prod(r - finite)), compute_lagrange_basis(r, finite)]
    )


class SmallValueRound(Round):
    """One accumulator round i: sᵢ = lᵢ · (Rᵢ · Aᵢ), then grow R by the round's
    Lagrange tensor and advance the eq mass. Bound to its round index and table.

    Deliberately no __call__: the phase-1 driver in prove_eq_poly_small_value is a
    manual loop over _round_poly / _fold, not fold_rounds, because the transition
    round rebuilds the bound-prefix eq table from the raw phase-1 challenges — which
    a Round.__call__ (returning only state/transcript/msg) would swallow."""

    def __init__(self, d: int, w_i: Array, accumulator: Array) -> None:
        self.d = d
        self.w_i = w_i
        self.accumulator = accumulator

    def _round_poly(self, state: SmallValueState) -> Array:
        r_tensor, eq_w_prev = state
        t_evals = (r_tensor[:, None] * self.accumulator).sum(axis=0)
        l_evals = expand_hypercube_step(eq_w_prev, self.w_i)
        return sumcheck_poly_from_t(t_evals, l_evals, self.d)

    def _fold(self, state: SmallValueState, r: Array) -> SmallValueState:
        r_tensor, eq_w_prev = state
        new_r = (
            r_tensor[:, None] * _lagrange_over_round_domain(r, self.d)[None, :]
        ).reshape(-1)
        return new_r, eq_w_prev * eq_factor(r, self.w_i)


def _precompute(p_initial: Array, w: Array, l_0: int) -> tuple[list[Array], Array]:
    """Accumulators A_1..A_{l₀} and the factors with eq(w,·) appended as the
    (d+1)-th factor (the transition round folds all d+1 together)."""
    l = log2_strict_usize(p_initial.shape[1])
    l_half = l // 2
    one = jnp.ones((), dtype=p_initial.dtype)

    e_in = expand_eq_to_hypercube(w[l_0 : l_0 + l_half], one)
    w_x_out = w[l_half + l_0 :]
    x_out_size = 1 << (l - l_half - l_0)
    e_out = [
        jnp.reshape(
            expand_eq_to_hypercube(jnp.concatenate([w[i:l_0], w_x_out]), one),
            (1 << (l_0 - i), x_out_size),
        )
        for i in range(1, l_0 + 1)
    ]
    accumulators = precompute_accumulators(p_initial, e_in, e_out)
    p_with_weights = jnp.vstack([p_initial, expand_eq_to_hypercube(w, one)[None, :]])
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

    # Phase 1: accumulator rounds. The manual loop captures the raw challenges the
    # transition needs to rebuild the bound-prefix eq table.
    state: SmallValueState = (jnp.ones(1, dtype=p_initial.dtype),) * 2
    challenges, msgs = [], []
    for i in range(1, l_0 + 1):
        rnd = SmallValueRound(d, w[i - 1], accumulators[i - 1])
        msg = rnd._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = rnd._fold(state, r[0])
        challenges.append(r[0])
        msgs.append(msg)
    eq_w_prev = state[1]

    # Phase 2: transition — refold all d+1 factors over the l₀ bound variables, send
    # the round poly (sliced to the d real factors), then fold to the tail state.
    eq_evals = jnp.ones(1, dtype=p_initial.dtype)
    for r in challenges:
        eq_evals = expand_hypercube_step(eq_evals, r)
    folded = compute_folded_evaluations(p_with_weights, eq_evals)
    msg_t = product_round_poly(folded)[:d]
    transcript, r_t = transcript.observe_and_sample(msg_t, 1)
    r_t = r_t[0]
    eq_w_prev = eq_w_prev * eq_factor(r_t, w[l_0])
    folded_p = fold_stacked(folded[:d], r_t)

    # Phase 3: the ordinary eq-poly tail. Product-bound: the accumulator precompute
    # (Procedure 9) contracts a product, so this engine is a product sumcheck only —
    # unlike EqPolyRound / SqrtSpaceRound, it does not take a general summand.
    (p_final, _), transcript, tail = fold_rounds(
        EqPolyRound(SumcheckRound(degree=d), w),
        (folded_p, eq_w_prev),
        transcript,
        l - l_0 - 1,
    )
    return p_final, transcript, msgs + [msg_t] + tail
