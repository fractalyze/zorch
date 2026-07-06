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
from zorch.transcript import Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.round import InnerVerifierRound


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Verifier for any sumcheck round; the dual of `prover.SumcheckRound`."""

    degree: int

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
        transcript, r = transcript.observe_and_sample(msg, 1)
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


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
@dataclass(frozen=True)
class CompressedCoeffsSumcheckRound(Round):
    """Verifier for the compressed coefficient wire
    (`prover.CompressedProductRound`): the message carries `[c_0, c_2]` of the
    degree-2 round polynomial; the linear coefficient never rides the wire and
    is reconstructed from the running claim — `s(1) = claim - s(0)` with
    `s(0) = c_0`, so `c_1 = s(1) - c_0 - c_2`. The reconstruction consumes the
    `s(0) + s(1) == claim` identity, so there is no per-round redundancy left to
    check (`ok` is constant true); binding rests on the terminal claim check,
    the trade the compressed form makes for wire size."""

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != 2:
            raise ValueError(
                f"compressed round message must carry [c_0, c_2], got shape "
                f"{msg.shape}"
            )
        c0, c2 = msg[0], msg[1]
        c1 = claim - c0 - c0 - c2  # s(1) - c_0 - c_2, with s(1) = claim - c_0
        transcript, r = transcript.observe_and_sample(msg, 1)
        reduced = eval_coeffs(jnp.stack([c0, c1, c2]), r[0])
        return reduced, transcript, r[0], jnp.bool_(True)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _eval_form: type[InnerVerifierRound] = SumcheckRound
    _coeffs_form: type[InnerVerifierRound] = CoeffsSumcheckRound
    _compressed_coeffs_form: type[InnerVerifierRound] = CompressedCoeffsSumcheckRound
