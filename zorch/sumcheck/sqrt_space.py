# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SqrtSpace sumcheck (Algorithm 2): a sumcheck proving the same round polynomials
as the linear-time prover while holding only O(√N) folded state.

The first l/2 rounds never fold the factors — they keep the running eq(r[<i], ·)
table and refold over the bound prefix on the fly (Formula 10, √-space for
recompute). The tail is a plain sumcheck (summand_evals + fold) over the state
folded down at the phase boundary. Messages match a linear-time prover on the same
challenges (testing/sqrt_space_test.py).

Generic over two orthogonal axes: the round summand (the SumcheckSummand seam
StandardRound and ProductSummand share) and the sampling EvalDomain. The √-space
memory trick
refolds each factor independently, so it is orthogonal to how the factors combine
AND to where the round poly is sampled. Defaults keep a product sumcheck on the
compressed Û_degree domain, so prove_sqrt_space(p) is unchanged; pass a homogeneous
summand and/or another EvalDomain (Gruen {0,1,*extra,eq_root(z)}, {0,½}, {0,2,4}) to
retarget it. The ∞-leading Û message needs a homogeneous summand (leading coeff =
combine(*slopes)); a finite domain lifts that restriction for any summand.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import expand_hypercube_step
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import EvalDomain, summand_evals, uhat_domain
from zorch.sumcheck.prover import (
    ProductSummand,
    StandardRound,
    SumcheckSummand,
    challenge_limbs,
)
from zorch.transcript import Transcript, sample_challenge
from zorch.utils.bits import log2_strict_usize

# (P_stacked, eq_evals): the factors (d, 2ˡ), untouched through phase 1, and the
# eq(r[<i], b) table over the bound prefix, b ∈ {0,1}ⁱ.
SqrtSpaceState = tuple[Array, Array]


def compute_folded_evaluations(p_stacked: Array, eq_evals: Array) -> Array:
    """Refold the factors over the bound prefix (Equation 4):
    pₖ(r[<i], x') = Σ_b eq(r[<i], b)·pₖ(b, x'), returning (d, 2ˡ⁻ⁱ)."""
    d = p_stacked.shape[0]
    l = log2_strict_usize(p_stacked.shape[1])
    i = log2_strict_usize(eq_evals.shape[0])
    p_reshaped = jnp.reshape(p_stacked, (d, 1 << i, 1 << (l - i)))
    return (p_reshaped * eq_evals[None, :, None]).sum(axis=1)


class SqrtSpaceRound(Round):
    """One first-phase round: refold on the fly, send the summand's round poly over
    the sampling domain, and extend the eq table by the sampled challenge (the
    factors stay put). Bound to a SumcheckSummand and an EvalDomain.

    `ext_dtype` chooses the challenge field exactly as `StandardRound` does — None
    keeps the byte-identical base-field squeeze, an extension squeezes its limbs — so a
    √-space run composes as the tail of an extension-field sumcheck (the univariate
    skip)."""

    def __init__(
        self,
        summand: SumcheckSummand,
        domain: EvalDomain,
        ext_dtype: Any | None = None,
    ) -> None:
        self.summand = summand
        self.domain = domain
        self.ext_dtype = ext_dtype
        self.limbs = 1 if ext_dtype is None else challenge_limbs(ext_dtype)

    def _round_poly(self, state: SqrtSpaceState) -> Array:
        return summand_evals(
            compute_folded_evaluations(*state), self.summand._combine, self.domain
        )

    def __call__(
        self, state: SqrtSpaceState, transcript: Transcript
    ) -> tuple[SqrtSpaceState, Transcript, Array]:
        p_stacked, eq_evals = state
        msg = self._round_poly(state)
        if self.ext_dtype is None:
            transcript, r = transcript.observe_and_sample(msg, 1)
            r = r[0]
        else:
            transcript = transcript.observe(msg)
            transcript, r = sample_challenge(transcript, self.ext_dtype, self.limbs)
        return (p_stacked, expand_hypercube_step(eq_evals, r)), transcript, msg


def prove_sqrt_space(
    p_initial: Array,
    transcript: Transcript,
    summand: SumcheckSummand | None = None,
    domain: EvalDomain | None = None,
    ext_dtype: Any | None = None,
) -> tuple[Array, Transcript, list[Array]]:
    """Prove the sumcheck: √-space first phase, standard second phase. `summand`
    defaults to the product over the factors (ProductSummand) and `domain` to the
    compressed Û_degree; pass a homogeneous SumcheckSummand and/or another EvalDomain
    to retarget the engine. `ext_dtype` chooses the challenge field (None = base, the
    byte-identical original; an extension makes this a drop-in tail for the univariate
    skip). Returns the final folded factors (d, 1), the transcript, and all l
    messages."""
    summand = summand or ProductSummand(degree=p_initial.shape[0])
    domain = domain or uhat_domain(summand.degree, p_initial.dtype)
    l = log2_strict_usize(p_initial.shape[1])
    l_half = l // 2

    state: SqrtSpaceState = (p_initial, jnp.ones(1, dtype=p_initial.dtype))
    state, transcript, phase1 = fold_rounds(
        SqrtSpaceRound(summand, domain, ext_dtype), state, transcript, l_half
    )

    # Boundary: materialize the deferred state — fold the whole bound prefix down at
    # once — then run standard sumcheck rounds over the explicit evaluations.
    folded = compute_folded_evaluations(*state)
    folded, transcript, phase2 = fold_rounds(
        StandardRound(summand, domain, ext_dtype=ext_dtype),
        folded,
        transcript,
        l - l_half,
    )
    return folded, transcript, phase1 + phase2
