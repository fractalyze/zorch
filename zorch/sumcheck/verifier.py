# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck verifier rounds -- the per-variable duals of the prover family.

`SumcheckRound` checks the round-poly identity and reduces the claim. It is
summand-agnostic: it sees only the round polynomials, so one verifier serves
every prover summand (product, LogUp, ...) at a given `degree`. The
observe -> sample order matches `prover.SumcheckRound.__call__` exactly, so
the prover's and verifier's Fiat-Shamir transcripts cannot diverge.

`CoeffsSumcheckRound` is the same check for a prover that sends ascending
coefficients instead of natural-domain values -- the wire form of a round
interpolated off a non-natural node set (e.g. through an eq factor's root),
where value form would force the verifier to know the sender's nodes. It also
owns the challenge squeeze rule (`challenge_limbs`), since a coefficient
prover's claims may live in an extension of the transcript's field.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.poly.univariate import eval_coeffs, eval_univariate
from zorch.round import Round
from zorch.sumcheck.prover import observe_and_sample_msg
from zorch.transcript import Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.round import InnerVerifierRound


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "scalar_framing"],
)
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Verifier for any sumcheck round; the dual of `prover.SumcheckRound`.

    `scalar_framing` must match the prover round's, so the two absorb the round
    poly under the same transcript framing and their Fiat-Shamir streams stay in
    sync (the shared `observe_and_sample_msg` is the one framing definition)."""

    degree: int
    scalar_framing: bool = False

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} evals, "
                f"got {msg.shape[0]}"
            )
        ok = claim == msg[0] + msg[1]
        transcript, r = observe_and_sample_msg(
            transcript, msg, 1, scalar_framing=self.scalar_framing
        )
        return eval_univariate(msg, r[0]), transcript, r[0], ok


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "challenge_limbs"],
)
@dataclass(frozen=True)
class CoeffsSumcheckRound(Round):
    """Verifier for a coefficient-form sumcheck round: `s(0) = c_0` and
    `s(1) = sum(c)`, so the identity check and the claim reduction read the
    coefficients directly."""

    degree: int
    challenge_limbs: int = 1

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.challenge_limbs < 1:
            raise ValueError("challenge_limbs must be >= 1")

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} "
                f"coefficients, got {msg.shape[0]}"
            )
        ok = claim == msg[0] + jnp.sum(msg)
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, claim.dtype, self.challenge_limbs)
        return eval_coeffs(msg, r), transcript, r, ok


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "scalar_framing"],
)
@dataclass(frozen=True)
class InfDomainSumcheckRound(Round):
    """Verifier for a round that sends ``(s(1), s(∞))`` — the value at 1 and the
    leading coefficient — the ∞-trick wire form paired with
    ``prover.SumcheckRound(domain=(1, INF))``. The running claim closes the third
    constraint: ``s(0) = claim − s(1)``, so the degree-2 poly reconstructs from
    ``{0, 1, ∞}`` division-free (``c0 = s(0)``, ``c2 = s(∞)``,
    ``c1 = s(1) − c0 − c2``). Like ``CoeffsSumcheckRound`` there is no independent
    per-round identity — the claim closes the missing value, so soundness is the
    end-to-end reduction. Degree 2 only (the product-of-two-MLEs round).

    `scalar_framing` must match the prover round's (see ``SumcheckRound``)."""

    degree: int
    scalar_framing: bool = False

    def __post_init__(self) -> None:
        if self.degree != 2:
            raise ValueError("the (1, INF) ∞-trick wire form is degree 2")

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != 2:
            raise ValueError(
                f"(s(1), s(inf)) message must have 2 entries, got {msg.shape[0]}"
            )
        s1, s_inf = msg[0], msg[1]
        s0 = claim - s1  # claim = s(0) + s(1)
        c1 = s1 - s0 - s_inf
        transcript, r = observe_and_sample_msg(
            transcript, msg, 1, scalar_framing=self.scalar_framing
        )
        r0 = r[0]
        reduced = s0 + c1 * r0 + s_inf * (r0 * r0)  # s(r0)
        return reduced, transcript, r0, jnp.array(True)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _eval_form: type[InnerVerifierRound] = SumcheckRound
    _coeffs_form: type[InnerVerifierRound] = CoeffsSumcheckRound
    _inf_form: type[InnerVerifierRound] = InfDomainSumcheckRound
