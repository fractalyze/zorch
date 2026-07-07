# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SqrtSpace sumcheck (Algorithm 2): a product sumcheck proving the same round
polynomials as the linear-time prover while holding only O(√N) folded state.

The first l/2 rounds never fold the factors — they keep the running eq(r[<i], ·)
table and refold over the bound prefix on the fly (Formula 10, √-space for
recompute). The tail is a plain product sumcheck (product_round_poly + fold) over
the state folded down at the phase boundary. Messages match a linear-time product
prover on the same challenges (testing/sqrt_space_test.py).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import expand_hypercube_step
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import fold_stacked, product_round_poly
from zorch.transcript import Transcript
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
    """One first-phase round: refold on the fly, send the product round poly, and
    extend the eq table by the sampled challenge (the factors stay put)."""

    def _round_poly(self, state: SqrtSpaceState) -> Array:
        return product_round_poly(compute_folded_evaluations(*state))

    def __call__(
        self, state: SqrtSpaceState, transcript: Transcript
    ) -> tuple[SqrtSpaceState, Transcript, Array]:
        p_stacked, eq_evals = state
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return (p_stacked, expand_hypercube_step(eq_evals, r[0])), transcript, msg


def prove_sqrt_space(
    p_initial: Array, transcript: Transcript
) -> tuple[Array, Transcript, list[Array]]:
    """Prove the product sumcheck: √-space first phase, standard second phase.
    Returns the final folded factors (d, 1), the transcript, and all l messages."""
    l = log2_strict_usize(p_initial.shape[1])
    l_half = l // 2

    state: SqrtSpaceState = (p_initial, jnp.ones(1, dtype=p_initial.dtype))
    state, transcript, phase1 = fold_rounds(SqrtSpaceRound(), state, transcript, l_half)

    # Fold the factors down over the bound prefix, then a plain product sumcheck.
    folded = compute_folded_evaluations(*state)
    phase2 = []
    for _ in range(l - l_half):
        msg = product_round_poly(folded)
        transcript, r = transcript.observe_and_sample(msg, 1)
        folded = fold_stacked(folded, r[0])
        phase2.append(msg)
    return folded, transcript, phase1 + phase2
