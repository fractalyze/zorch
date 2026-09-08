# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""EqPoly sumcheck (Algorithm 5): an eq-weighted sumcheck of d multilinears against
an equality weight eq(w, ·), with eq factored into left/right suffixes so no round
materializes eq over the full hypercube.

Each round sends sᵢ = lᵢ · tᵢ sampled at the round's EvalDomain: tᵢ = Σₓ eq-weight ·
combine(folded factors), lᵢ the linear eq factor of the current variable. Both the
summand `combine` (SumcheckSummand — product by default) and the sampling domain
(the compressed Û_d = {∞, 0, 2, …, d−1} by default) are settable; a leading ∞ point
needs a homogeneous combine (see domain.summand_evals). The state width halves each
round, so a fixed-shape lax.scan does not fit: prove_eq_poly drives one EqPolyRound
through the fold_rounds host loop. Correctness anchor: the default (product, Û)
messages equal a plain product sumcheck over [P₁, …, P_d, eq(w,·)]
(testing/eq_poly_test.py).
"""

from __future__ import annotations

from collections.abc import Callable

import frx
import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.poly.eq import eq_factor, expand_eq_family, expand_hypercube_step
from zorch.prove import fold_rounds
from zorch.round import ProverRound, RunningClaim
from zorch.sumcheck.domain import (
    EvalDomain,
    split_halves,
    split_pairs,
    uhat_domain,
)
from zorch.sumcheck.prover import FoldingClaim, ProductSummand, SumcheckSummand
from zorch.sumcheck.reduce import reduce_domain
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.utils.field import naturals

# (P_stacked, eq_w_prev): the d folded factors (d, 2ˡ⁻ⁱ) and the running eq mass
# eq(w[<i], r[<i]) as a length-1 array.
EqPolyState = tuple[Array, Array]


def compute_eq_evaluations(w: Array) -> list[Array]:
    """Suffix eq tables [eq(w[-1:], ·), …, eq(w, ·)], entry i over {0,1}ⁱ⁺¹.

    Scans w backwards, prepending each coordinate as the MSB; large members are
    emitted outer-split (see `expand_eq_family`)."""
    return expand_eq_family(w, msb=True, suffix=True)


def compute_eq_prefixes(w: Array) -> list[Array]:
    """Prefix eq tables [eq(w[:1], ·), …, eq(w, ·)], entry i over {0,1}ⁱ⁺¹.

    Scans w forwards, appending each coordinate as the LSB, so `w[0]` stays the
    MSB of every table. The dual of compute_eq_evaluations: a round that binds
    the LOW variable consumes w from the back, so what it has left is a prefix."""
    return expand_eq_family(w, msb=False, suffix=False)


def _split_slope(p_stacked: Array, *, msb: bool = True) -> tuple[Array, Array]:
    """Halve each factor on the current variable into (P0, slope P1−P0), each
    (d, N/2) — the split in the (P0, slope) form the eq round folds by. `msb`
    takes contiguous halves (`split_halves`), `msb=False` stride-2 pairs
    (`split_pairs`), matching domain.fold's two bind orders."""
    p0, p1 = split_halves(p_stacked) if msb else split_pairs(p_stacked)
    return p0, p1 - p0


def _weighted_summand(
    p0s: Array,
    diffs: Array,
    eq_w_l: Array | None,
    eq_w_r: Array,
    domain: EvalDomain,
    combine: Callable[..., Array],
) -> Array:
    """tᵢ sampled at `domain`: t(point) = Σₓ eq-weight(x) · combine(f₁, …, f_m)(x) with
    fₖ the linear factor (P0ₖ, P0ₖ+diffsₖ) sampled at that point (a leading ∞ point
    gives combine(*diffs), valid for a homogeneous combine — see domain.summand_evals).
    Early rounds carry both eq halves (eq_w_l set, factors reshaped into x_L × x_R);
    late rounds weight by eq_w_r. combine defaults to the product Πₖ fₖ."""
    m = p0s.shape[0]
    p1s = p0s + diffs
    if eq_w_l is not None:
        num_x_l = eq_w_l.shape[0]
        num_x_r = p0s.shape[1] // num_x_l
        p0s = fnp.reshape(p0s, (m, num_x_l, num_x_r))
        p1s = fnp.reshape(p1s, (m, num_x_l, num_x_r))
    combined = combine(*frx.vmap(domain.sample)(p0s, p1s))  # (num_points, *x_shape)
    if eq_w_l is not None:
        return fnp.sum(
            eq_w_l[None, :, None] * combined * eq_w_r[None, None, :], axis=(1, 2)
        )
    return fnp.sum(combined * eq_w_r[None, :], axis=1)


def sumcheck_poly_from_t(t_evals: Array, l_evals: Array, domain: EvalDomain) -> Array:
    """sᵢ = lᵢ · tᵢ sampled at `domain`: at a leading ∞ point s(∞) = l_diff·t(∞); at a
    finite node s(node) = (l(0) + node·l_diff)·t(node), l the linear eq factor of the
    round variable. Same body for the compressed Û message and the full coeff domain —
    they differ only in `domain`."""
    l_0, l_1 = l_evals[0], l_evals[1]
    l_diff = l_1 - l_0
    finite_t = t_evals[1:] if domain.inf_index is not None else t_evals
    finite = (l_0 + domain.nodes * l_diff) * finite_t
    if domain.inf_index is not None:
        return fnp.concatenate([fnp.atleast_1d(l_diff * t_evals[0]), finite])
    return finite


class EqPolyRound(ProverRound):
    """One EqPoly variable-binding round, reused across all l rounds — it reads the
    round index off the state width, so one object drives the whole proof. Bound to a
    homogeneous SumcheckSummand (its combine weighted by eq); product by default."""

    def __init__(
        self,
        summand: SumcheckSummand,
        w: Array,
        domain: EvalDomain | None = None,
        *,
        challenges: ChallengePolicy,
        msb: bool = True,
    ) -> None:
        self.summand = summand
        self.w = w
        self.domain = domain or uhat_domain(summand.degree, w.dtype)
        self.challenges = challenges
        self.msb = msb
        self.l = int(w.shape[0])
        self.l_half = self.l // 2
        tables = compute_eq_evaluations if msb else compute_eq_prefixes
        self.eq_w_l_list = tables(w[: self.l_half])
        self.eq_w_r_list = tables(w[self.l_half :])

    def _w_index(self, i: int) -> int:
        """The coordinate round `i` binds. Binding the high variable walks w
        forwards from w[0]; binding the low variable walks it backwards."""
        return i - 1 if self.msb else self.l - i

    def _eq_tables(self, p_stacked: Array) -> tuple[int, Array | None, Array]:
        """Round index i and the eq weights for it: both halves early, one alone
        late.

        The unbound cube stays MSB-indexed either way, so `eq_w_l` always weights
        its high part and `eq_w_r` its low part. What the bind order changes is
        *which* coordinates survive — binding high leaves a suffix of w, binding
        low leaves a prefix — so the two directions read their tables from
        opposite ends.
        """
        i = self.l - log2_strict_usize(p_stacked.shape[1]) + 1
        rest = self.l - i  # coordinates this round leaves unbound
        if rest == 0:
            # The final round weights nothing. Worth stating rather than falling
            # through to a negative index: Σₓ eq(w, x) = 1, so summing against a
            # whole table would also scale by one, but only by accident.
            return i, None, fnp.ones(1, dtype=self.w.dtype)
        if self.msb:
            if i < self.l_half:
                return i, self.eq_w_l_list[(self.l_half - i) - 1], self.eq_w_r_list[-1]
            return i, None, self.eq_w_r_list[rest - 1]
        if rest > self.l_half:
            return i, self.eq_w_l_list[-1], self.eq_w_r_list[(rest - self.l_half) - 1]
        return i, None, self.eq_w_l_list[rest - 1]

    def _round_poly(
        self, state: EqPolyState
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """The round message sampled at self.domain (the compressed Û_degree by
        default) — the oracle anchor __call__ binds Fiat-Shamir to (the standalone
        coefficient form is _round_coeffs)."""
        p_stacked, eq_w_prev = state
        i, eq_w_l, eq_w_r = self._eq_tables(p_stacked)
        p0s, diffs = _split_slope(p_stacked, msb=self.msb)
        t_evals = _weighted_summand(
            p0s, diffs, eq_w_l, eq_w_r, self.domain, self.summand._combine
        )
        w_i = self.w[self._w_index(i)]
        l_evals = expand_hypercube_step(eq_w_prev, w_i)
        return sumcheck_poly_from_t(t_evals, l_evals, self.domain), (
            p0s,
            diffs,
            w_i,
        )

    def _round_coeffs(
        self, state: EqPolyState
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Ascending coefficients of the degree-(degree+1) round polynomial sᵢ = lᵢ·tᵢ.
        Sampled on the full round domain [∞, 0, 1, …, degree] (u=1 kept) so s is fully
        determined — unlike the compressed self.domain, this is standalone-verifiable.
        """
        p_stacked, eq_w_prev = state
        degree = self.summand.degree
        i, eq_w_l, eq_w_r = self._eq_tables(p_stacked)
        p0s, diffs = _split_slope(p_stacked, msb=self.msb)
        w_i = self.w[self._w_index(i)]
        full = EvalDomain(naturals(degree + 1, p_stacked.dtype), inf_index=0)
        t = _weighted_summand(p0s, diffs, eq_w_l, eq_w_r, full, self.summand._combine)
        l_evals = expand_hypercube_step(eq_w_prev, w_i)  # lᵢ(0), lᵢ(1)
        s = sumcheck_poly_from_t(t, l_evals, full)
        coeffs = EvalDomain(inf_index=0).to_coeffs(s)
        return coeffs, (p0s, diffs, w_i)

    def _fold(
        self, cache: tuple[Array, Array, Array], eq_w_prev: Array, r: Array
    ) -> EqPolyState:
        p0s, diffs, w_i = cache
        return diffs * r + p0s, eq_w_prev * eq_factor(r, w_i)

    def __call__(
        self, carry: FoldingClaim, transcript: Transcript
    ) -> tuple[FoldingClaim, Transcript, Array]:
        msg, cache = self._round_poly(carry.state)
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        reduced, _ = reduce_domain(carry.claim.value, msg, r, self.domain)
        folded = self._fold(cache, carry.state[1], r)
        return carry.advance(folded, reduced, r), transcript, msg


def prove_eq_poly(
    p_initial: Array,
    w: Array,
    claim: Array,
    transcript: Transcript,
    summand: SumcheckSummand | None = None,
    domain: EvalDomain | None = None,
    *,
    challenges: ChallengePolicy,
    msb: bool = True,
) -> tuple[Array, Transcript, list[Array]]:
    """Fold all l variables; return the final factors (d, 1), the advanced
    transcript, and the per-round messages (each sᵢ over Û_d).

    Fiat-Shamir binds to the compressed Û_d message, which drops u=1 and so is not
    standalone-verifiable. The standalone coefficient form is EqPolyRound._round_coeffs,
    checked round-by-round against verifier.CoeffsSumcheckRound — a distinct transcript,
    not a re-encoding of the Û_d proof returned here."""
    rounds = log2_strict_usize(p_initial.shape[1])
    if w.shape[0] != rounds:
        raise ValueError(
            f"w needs one weight per variable: got {w.shape[0]} for {rounds} variables"
        )
    start = RunningClaim(claim, fnp.zeros((rounds,), challenges.dtype), fnp.int32(0))
    rnd = EqPolyRound(
        summand or ProductSummand(degree=p_initial.shape[0]),
        w,
        domain,
        challenges=challenges,
        msb=msb,
    )
    state: EqPolyState = (p_initial, fnp.ones(1, dtype=p_initial.dtype))
    carry, transcript, msgs = fold_rounds(
        rnd, FoldingClaim(state, start), transcript, rounds
    )
    return carry, transcript, msgs
