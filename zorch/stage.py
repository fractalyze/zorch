# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable roles for conditional proof reductions.

A stage is one mathematical claim-reduction contract. Its prover and verifier
are separate runtime objects so role-specific capabilities, especially proving
and verification keys, never need to share an object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from frx import Array

from zorch.transcript import Transcript

Claim = TypeVar("Claim")
Witness = TypeVar("Witness")
ReducedClaim = TypeVar("ReducedClaim")
ReductionProof = TypeVar("ReductionProof")


@dataclass(frozen=True)
class TrivialClaim:
    """The claim that holds by construction, so nothing remains to prove.

    A stage reducing to this is a complete argument rather than one link in a
    chain: an argument of knowledge is exactly a reduction to the trivial
    claim.
    """


@dataclass(frozen=True)
class ProveResult(Generic[ReducedClaim, ReductionProof]):
    """A reduced claim, its conditional reduction proof, and the transcript."""

    reduced_claim: ReducedClaim
    reduction_proof: ReductionProof
    transcript: Transcript


@dataclass(frozen=True)
class VerifyResult(Generic[ReducedClaim]):
    """The verifier-derived reduced claim, advanced transcript, and verdict."""

    reduced_claim: ReducedClaim
    transcript: Transcript
    ok: Array


class ProverStage(ABC, Generic[Claim, Witness, ReducedClaim, ReductionProof]):
    """The prover role of one conditional claim reduction."""

    @abstractmethod
    def prove(
        self, claim: Claim, witness: Witness, transcript: Transcript
    ) -> ProveResult[ReducedClaim, ReductionProof]: ...


class VerifierStage(ABC, Generic[Claim, ReducedClaim, ReductionProof]):
    """The verifier role of the same conditional claim reduction."""

    @abstractmethod
    def verify(
        self,
        claim: Claim,
        reduction_proof: ReductionProof,
        transcript: Transcript,
    ) -> VerifyResult[ReducedClaim]: ...


@dataclass(frozen=True)
class Stage(Generic[Claim, Witness, ReducedClaim, ReductionProof]):
    """Optional pairing of separately deployable roles for tests and tooling.

    Deployment code should depend directly on ``ProverStage`` or
    ``VerifierStage`` and therefore construct only the capabilities it owns.
    """

    prover: ProverStage[Claim, Witness, ReducedClaim, ReductionProof]
    verifier: VerifierStage[Claim, ReducedClaim, ReductionProof]
