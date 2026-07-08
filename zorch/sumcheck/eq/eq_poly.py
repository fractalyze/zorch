# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""EqPoly sumcheck (Algorithm 5): product sumcheck of d multilinears against an
equality weight eq(w, ·), with eq factored into left/right suffixes so no round
materializes eq over the full hypercube.

Each round sends sᵢ = lᵢ · tᵢ over Û_d = {∞, 0, 2, …, d−1} (a bare Array; ∞ is the
leading coefficient, u=1 is omitted and recovered by the verifier from
s(0)+s(1)=claim). tᵢ is the degree-d product of the folded factors; lᵢ is the
linear eq factor of the current variable. The state width halves each round, so a
fixed-shape lax.scan does not fit: prove_eq_poly drives one EqPolyRound
through the fold_rounds host loop. Correctness anchor: the messages equal a plain
product sumcheck over [P₁, …, P_d, eq(w,·)] (testing/eq_poly_test.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import eq_factor, expand_hypercube_step
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import EvalDomain, _naturals
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

# (P_stacked, eq_w_prev): the d folded factors (d, 2ˡ⁻ⁱ) and the running eq mass
# eq(w[<i], r[<i]) as a length-1 array.
EqPolyState = tuple[Array, Array]


def compute_eq_evaluations(w: Array) -> list[Array]:
    """Suffix eq tables [eq(w[-1:], ·), …, eq(w, ·)], entry i over {0,1}ⁱ.

    Scans w backwards, prepending each coordinate as the MSB."""
    v = jnp.ones(1, dtype=w.dtype)
    v_list = []
    for i in range(w.shape[0] - 1, -1, -1):
        v = expand_hypercube_step(v, w[i], msb=True)
        v_list.append(v)
    return v_list


def _split_pairs(p_stacked: Array) -> tuple[Array, Array]:
    """Halve each factor on the current variable into (P0, P1−P0), each (d, N/2)."""
    d = p_stacked.shape[0]
    pairs = jnp.reshape(p_stacked, (d, 2, -1))
    p0 = pairs[:, 0, :]
    return p0, pairs[:, 1, :] - p0


def _weighted_product(
    p0s: Array, diffs: Array, eq_w_l: Array | None, eq_w_r: Array, us: Sequence[int]
) -> Array:
    """[t(∞), t(u) for u in us] where t(u) = Σₓ eq-weight(x) · Πₖ (diffsₖ·u + P0ₖ)
    and t(∞) is the leading coefficient Πₖ diffsₖ. Early rounds carry both eq halves
    (eq_w_l set, factors reshaped into x_L × x_R); late rounds weight by eq_w_r."""
    d = p0s.shape[0]
    if eq_w_l is not None:
        num_x_l = eq_w_l.shape[0]
        num_x_r = p0s.shape[1] // num_x_l
        p0s = jnp.reshape(p0s, (d, num_x_l, num_x_r))
        diffs = jnp.reshape(diffs, (d, num_x_l, num_x_r))

    def weighted_sum(vals: Array) -> Array:
        if eq_w_l is not None:
            return jnp.sum(eq_w_l[:, None] * vals * eq_w_r[None, :])
        return jnp.sum(vals * eq_w_r)

    evals = [weighted_sum(jnp.prod(diffs, axis=0))]
    for u in us:
        evals.append(weighted_sum(jnp.prod(diffs * u + p0s, axis=0)))
    return jnp.stack(evals)


def compute_t_poly(
    p0s: Array, diffs: Array, eq_w_l: Array | None, eq_w_r: Array
) -> Array:
    """tᵢ over Û_d = [t(∞), t(0), t(2), …, t(d−1)] — the compressed form the round
    message oracle checks against."""
    d = p0s.shape[0]
    return _weighted_product(p0s, diffs, eq_w_l, eq_w_r, [0, *range(2, d)])


def _mul_linear_into_t(t_evals: Array, l_evals: Array, us: Array) -> Array:
    """s(u) = l(u) · t(u) node-wise over [∞, *us], where l is the linear eq factor
    l(u) = l(0) + u·(l(1)−l(0)): s(∞) = l_diff·t(∞), s(u) = (l0 + u·l_diff)·t(u).
    The compressed Û_d message and the full-domain coefficient form differ only in
    the finite node set us, so both route through here."""
    l_0, l_1 = l_evals[0], l_evals[1]
    l_diff = l_1 - l_0
    return jnp.concatenate(
        [jnp.atleast_1d(l_diff * t_evals[0]), (l_0 + us * l_diff) * t_evals[1:]]
    )


def sumcheck_poly_from_t(t_evals: Array, l_evals: Array, d: int) -> Array:
    """sᵢ = lᵢ · tᵢ over Û_d, shape (d,). l_evals = [l(0), l(1)]; the product is
    pointwise per node, so s(∞) is the product of leading coefficients."""
    nat = _naturals(d, t_evals.dtype)  # [0, 1, …, d−1]
    us = jnp.concatenate([nat[:1], nat[2:]])  # Û_d finite nodes: u=1 dropped
    return _mul_linear_into_t(t_evals, l_evals, us)


class EqPolyRound(Round):
    """One EqPoly variable-binding round, reused across all l rounds — it reads the
    round index off the state width, so one object drives the whole proof."""

    def __init__(self, d: int, w: Array) -> None:
        self.d = d
        self.w = w
        self.l = int(w.shape[0])
        self.l_half = self.l // 2
        self.eq_w_l_list = compute_eq_evaluations(w[: self.l_half])
        self.eq_w_r_list = compute_eq_evaluations(w[self.l_half :])

    def _eq_tables(self, p_stacked: Array) -> tuple[int, Array | None, Array]:
        """Round index i and the eq weights for it: both halves early (i < l/2),
        the right half alone late."""
        i = self.l - log2_strict_usize(p_stacked.shape[1]) + 1
        if i < self.l_half:
            return i, self.eq_w_l_list[(self.l_half - i) - 1], self.eq_w_r_list[-1]
        return i, None, self.eq_w_r_list[(self.l - i) - 1]

    def _round_poly(
        self, state: EqPolyState
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """The compressed Û_d round message [sᵢ(∞), sᵢ(0), sᵢ(2), …] — the oracle
        anchor; __call__ sends the coefficient form."""
        p_stacked, eq_w_prev = state
        i, eq_w_l, eq_w_r = self._eq_tables(p_stacked)
        p0s, diffs = _split_pairs(p_stacked)
        t_evals = compute_t_poly(p0s, diffs, eq_w_l, eq_w_r)
        l_evals = expand_hypercube_step(eq_w_prev, self.w[i - 1])
        return sumcheck_poly_from_t(t_evals, l_evals, self.d), (
            p0s,
            diffs,
            self.w[i - 1],
        )

    def _round_coeffs(
        self, state: EqPolyState
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Ascending coefficients of the degree-(d+1) round polynomial sᵢ = lᵢ · tᵢ.
        tᵢ is taken at the full round domain [∞, 0, 1, …, d] so s is fully determined
        (the compressed Û_d form drops a point and is not standalone-verifiable)."""
        p_stacked, eq_w_prev = state
        i, eq_w_l, eq_w_r = self._eq_tables(p_stacked)
        p0s, diffs = _split_pairs(p_stacked)
        w_i = self.w[i - 1]
        t = _weighted_product(p0s, diffs, eq_w_l, eq_w_r, range(self.d + 1))
        l_evals = expand_hypercube_step(eq_w_prev, w_i)  # lᵢ(0), lᵢ(1)
        # Full round domain [∞, 0, 1, …, d] (u=1 kept) so s is standalone-verifiable.
        s = _mul_linear_into_t(t, l_evals, _naturals(self.d + 1, t.dtype))
        coeffs = EvalDomain(leading=True).to_coeffs(s)
        return coeffs, (p0s, diffs, w_i)

    def _fold(
        self, cache: tuple[Array, Array, Array], eq_w_prev: Array, r: Array
    ) -> EqPolyState:
        p0s, diffs, w_i = cache
        return diffs * r + p0s, eq_w_prev * eq_factor(r, w_i)

    def __call__(
        self, state: EqPolyState, transcript: Transcript
    ) -> tuple[EqPolyState, Transcript, Array]:
        msg, cache = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return self._fold(cache, state[1], r[0]), transcript, msg


def prove_eq_poly(
    p_initial: Array, w: Array, transcript: Transcript
) -> tuple[Array, Transcript, list[Array]]:
    """Fold all l variables; return the final factors (d, 1), the advanced
    transcript, and the per-round messages (each sᵢ over Û_d).

    Fiat-Shamir binds to the compressed Û_d message (the byte-match anchor to the
    whir-zorch/SP1 on-wire form), which drops u=1 and is not standalone-verifiable —
    its driver-level verifier dual is pending (issue #410). The independent,
    standalone-verifiable coefficient form is EqPolyRound._round_coeffs, checked
    round-by-round against verifier.CoeffsSumcheckRound (see test_coeff_round_verifies);
    it is a distinct transcript, not a re-encoding of the Û_d proof returned here."""
    rounds = log2_strict_usize(p_initial.shape[1])
    if w.shape[0] != rounds:
        raise ValueError(
            f"w needs one weight per variable: got {w.shape[0]} for {rounds} variables"
        )
    rnd = EqPolyRound(p_initial.shape[0], w)
    state: EqPolyState = (p_initial, jnp.ones(1, dtype=p_initial.dtype))
    (p_final, _), transcript, msgs = fold_rounds(rnd, state, transcript, rounds)
    return p_final, transcript, msgs
