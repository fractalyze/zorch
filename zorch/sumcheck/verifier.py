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
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array

from zorch.poly.univariate import eval_coeffs, eval_univariate
from zorch.round import Round
from zorch.sumcheck.domain import subgroup_sum
from zorch.transcript import Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.round import InnerVerifierRound


@partial(frx.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Verifier for any sumcheck round; the dual of `prover.SumcheckRound`."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def check_reduce(self, claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
        """The round identity + claim reduction alone, for an externally
        sampled challenge — the dual of `prover.SumcheckRound.round_poly`,
        so a driver whose choreography owns the Fiat-Shamir hop still routes
        the round math through one definition. Returns `(reduced, ok)`."""
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} evals, "
                f"got {msg.shape[0]}"
            )
        ok = claim == msg[0] + msg[1]
        return eval_univariate(msg, r), ok

    def __call__(
        self, claim: Array, transcript: Transcript, msg: Array
    ) -> tuple[Array, Transcript, tuple[Array, Array]]:
        transcript, r = transcript.observe_and_sample(msg, 1)
        reduced, ok = self.check_reduce(claim, msg, r[0])
        return reduced, transcript, (r[0], ok)


@partial(
    frx.tree_util.register_dataclass,
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
        self, claim: Array, transcript: Transcript, msg: Array
    ) -> tuple[Array, Transcript, tuple[Array, Array]]:
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} "
                f"coefficients, got {msg.shape[0]}"
            )
        ok = claim == msg[0] + fnp.sum(msg)
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, claim.dtype, self.challenge_limbs)
        return eval_coeffs(msg, r), transcript, (r, ok)


@partial(frx.tree_util.register_dataclass, data_fields=[], meta_fields=[])
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

    def check_reduce(self, claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
        """Reconstruct `c_1` from the claim and reduce, for an externally
        sampled challenge — mirrors `SumcheckRound.check_reduce`. `ok` is the
        constant true of the compressed form (the redundancy was spent on the
        reconstruction)."""
        if msg.shape[0] != 2:
            raise ValueError(
                f"compressed round message must carry [c_0, c_2], got shape "
                f"{msg.shape}"
            )
        c0, c2 = msg[0], msg[1]
        c1 = claim - c0 - c0 - c2  # s(1) - c_0 - c_2, with s(1) = claim - c_0
        return eval_coeffs(fnp.stack([c0, c1, c2]), r), fnp.bool_(True)

    def __call__(
        self, claim: Array, transcript: Transcript, msg: Array
    ) -> tuple[Array, Transcript, tuple[Array, Array]]:
        transcript, r = transcript.observe_and_sample(msg, 1)
        reduced, ok = self.check_reduce(claim, msg, r[0])
        return reduced, transcript, (r[0], ok)


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["skip_rounds", "degree", "ext_dtype", "challenge_limbs"],
)
@dataclass(frozen=True)
class UnivariateSkipRound(Round):
    """Verifier for the univariate skip's round 0 (`sumcheck.univariate_skip`): the
    message is the round polynomial s₀ in ascending-coefficient form (degree
    `degree·(|D|−1)`, |D| = 2^skip_rounds), so the round identity is the SUBGROUP-sum
    check
    `c == Σ_{z∈D} s₀(z)` — `subgroup_sum` reads it off the coefficients at multiples
    of |D| — and the claim reduces to `s₀(r₀)`. The subgroup sibling of
    `CoeffsSumcheckRound`, whose hypercube identity (`s(0)=c₀`, `s(1)=Σc`) it swaps for
    the subgroup sum. r₀ ∈ F_ext, so the challenge takes `challenge_limbs` squeezes of
    `ext_dtype` (the reduced claim promotes to it via the base coefficients × r₀); a
    base-field run is `ext_dtype` prime with `challenge_limbs == 1`. `skip_rounds` is
    the number of collapsed leading rounds (|D| = 2^skip_rounds)."""

    skip_rounds: int
    degree: int
    ext_dtype: Any
    challenge_limbs: int = 1

    def __post_init__(self) -> None:
        if self.skip_rounds < 1:
            raise ValueError(
                "skip_rounds must be >= 1 (skip_rounds=0 is the plain sumcheck run)"
            )
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.challenge_limbs < 1:
            raise ValueError("challenge_limbs must be >= 1")

    def __call__(
        self, claim: Array, transcript: Transcript, msg: Array
    ) -> tuple[Array, Transcript, tuple[Array, Array]]:
        d0 = self.degree * ((1 << self.skip_rounds) - 1)
        if msg.shape[0] != d0 + 1:
            raise ValueError(
                f"round-0 message must have degree·(|D|−1)+1={d0 + 1} coefficients, "
                f"got {msg.shape[0]}"
            )
        ok = claim == subgroup_sum(msg, self.skip_rounds)
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(
            transcript, self.ext_dtype, self.challenge_limbs
        )
        return eval_coeffs(msg, r), transcript, (r, ok)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _eval_form: type[InnerVerifierRound] = SumcheckRound
    _coeffs_form: type[InnerVerifierRound] = CoeffsSumcheckRound
    _compressed_coeffs_form: type[InnerVerifierRound] = CompressedCoeffsSumcheckRound
    _univariate_skip_form: type[InnerVerifierRound] = UnivariateSkipRound
