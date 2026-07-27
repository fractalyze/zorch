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
where value form would force the verifier to know the sender's nodes. Like
all other sumcheck rounds, it receives one shared `ChallengePolicy`; coefficient
claims may therefore live in an extension of the transcript field.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.poly.univariate import eval_coeffs
from zorch.round import RunningClaim, VerifierRound
from zorch.sumcheck.domain import subgroup_sum
from zorch.sumcheck.reduce import (
    reduce_coeffs,
    reduce_compressed,
    reduce_evals,
    require_width,
)
from zorch.transcript import Transcript


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "challenges"],
)
@dataclass(frozen=True)
class SumcheckRound(VerifierRound):
    """Verifier for any sumcheck round; the dual of `prover.SumcheckRound`."""

    degree: int
    challenges: ChallengePolicy

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def check_reduce(self, claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
        """The round identity and claim reduction, for an externally sampled
        challenge. Both roles route their round math through
        `sumcheck.reduce`, so a driver that owns the Fiat-Shamir hop itself —
        the prover accumulating its own claim included — reduces identically."""
        return reduce_evals(claim, msg, r, self.degree)

    def __call__(
        self, claim: RunningClaim, transcript: Transcript, msg: Array
    ) -> tuple[RunningClaim, Transcript, Array]:
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        reduced, ok = self.check_reduce(claim.value, msg, r)
        return claim.bind(reduced, r), transcript, ok


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "challenges"],
)
@dataclass(frozen=True)
class CoeffsSumcheckRound(VerifierRound):
    """Verifier for a coefficient-form sumcheck round: `s(0) = c_0` and
    `s(1) = sum(c)`, so the identity check and the claim reduction read the
    coefficients directly."""

    degree: int
    challenges: ChallengePolicy

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def __call__(
        self, claim: RunningClaim, transcript: Transcript, msg: Array
    ) -> tuple[RunningClaim, Transcript, Array]:
        # Structural rejection precedes any read of the claim.
        require_width(msg, self.degree + 1, "coefficients")
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        reduced, ok = reduce_coeffs(claim.value, msg, r, self.degree)
        return claim.bind(reduced, r), transcript, ok


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["challenges"],
)
@dataclass(frozen=True)
class CompressedCoeffsSumcheckRound(VerifierRound):
    """Verifier for the compressed coefficient wire
    (`prover.CompressedProductRound`): the message carries `[c_0, c_2]` of the
    degree-2 round polynomial; the linear coefficient never rides the wire and
    is reconstructed from the running claim — `s(1) = claim - s(0)` with
    `s(0) = c_0`, so `c_1 = s(1) - c_0 - c_2`. The reconstruction consumes the
    `s(0) + s(1) == claim` identity, so there is no per-round redundancy left to
    check (`ok` is constant true); binding rests on the terminal claim check,
    the trade the compressed form makes for wire size."""

    challenges: ChallengePolicy

    def check_reduce(self, claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
        """Reconstruct `c_1` from the claim and reduce, for an externally
        sampled challenge — mirrors `SumcheckRound.check_reduce`."""
        return reduce_compressed(claim, msg, r)

    def __call__(
        self, claim: RunningClaim, transcript: Transcript, msg: Array
    ) -> tuple[RunningClaim, Transcript, Array]:
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        reduced, ok = self.check_reduce(claim.value, msg, r)
        return claim.bind(reduced, r), transcript, ok


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["skip_rounds", "degree", "challenges"],
)
@dataclass(frozen=True)
class UnivariateSkipRound(VerifierRound):
    """Verifier for the univariate skip's round 0 (`sumcheck.univariate_skip`): the
    message is the round polynomial s₀ in ascending-coefficient form (degree
    `degree·(|D|−1)`, |D| = 2^skip_rounds), so the round identity is the SUBGROUP-sum
    check
    `c == Σ_{z∈D} s₀(z)` — `subgroup_sum` reads it off the coefficients at multiples
    of |D| — and the claim reduces to `s₀(r₀)`. The subgroup sibling of
    `CoeffsSumcheckRound`, whose hypercube identity (`s(0)=c₀`, `s(1)=Σc`) it swaps for
    the subgroup sum. Its `ChallengePolicy` selects the target field and squeeze
    width; evaluating the base coefficients at that challenge promotes the reduced
    claim as needed. `skip_rounds` is
    the number of collapsed leading rounds (|D| = 2^skip_rounds)."""

    skip_rounds: int
    degree: int
    challenges: ChallengePolicy

    def __post_init__(self) -> None:
        if self.skip_rounds < 1:
            raise ValueError(
                "skip_rounds must be >= 1 (skip_rounds=0 is the plain sumcheck run)"
            )
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def __call__(
        self, claim: RunningClaim, transcript: Transcript, msg: Array
    ) -> tuple[RunningClaim, Transcript, Array]:
        d0 = self.degree * ((1 << self.skip_rounds) - 1)
        if msg.shape[0] != d0 + 1:
            raise ValueError(
                f"round-0 message must have degree·(|D|−1)+1={d0 + 1} coefficients, "
                f"got {msg.shape[0]}"
            )
        ok = claim.value == subgroup_sum(msg, self.skip_rounds)
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        return claim.bind(eval_coeffs(msg, r), r), transcript, ok


if TYPE_CHECKING:
    _SumcheckWire = VerifierRound[RunningClaim, Array]

    _eval_form: type[_SumcheckWire] = SumcheckRound
    _coeffs_form: type[_SumcheckWire] = CoeffsSumcheckRound
    _compressed_coeffs_form: type[_SumcheckWire] = CompressedCoeffsSumcheckRound
    _univariate_skip_form: type[_SumcheckWire] = UnivariateSkipRound
