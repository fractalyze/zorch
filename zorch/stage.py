# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable roles for conditional proof reductions.

A stage is one mathematical claim-reduction contract. Its prover and verifier
are separate runtime objects so role-specific capabilities, especially proving
and verification keys, never need to share an object.

The roles are structural, like `ProverRound`/`VerifierRound` and `Transcript`:
an adapter or a wrapper conforms by shape, without inheriting. Their members
are `@abstractmethod`, so a class that *does* subclass one fails loudly at
construction if it leaves a role method unimplemented — structural conformance
for foreign types, nominal enforcement for its own implementers.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from frx import Array

from zorch.transcript import TranscriptT

Claim = TypeVar("Claim")
Witness = TypeVar("Witness")
ReducedClaim = TypeVar("ReducedClaim")
ReductionProof = TypeVar("ReductionProof")

# Argument-position parameters are contravariant, which is the contract read
# out loud: both roles consume a claim, the prover consumes a witness, and the
# reduction proof the prover produces is the one the verifier consumes.
Claim_contra = TypeVar("Claim_contra", contravariant=True)
Witness_contra = TypeVar("Witness_contra", contravariant=True)
ReductionProof_contra = TypeVar("ReductionProof_contra", contravariant=True)


@dataclass(frozen=True)
class TrivialClaim:
    """The claim that holds by construction, so nothing remains to prove.

    A stage reducing to this is a complete argument rather than one link in a
    chain: an argument of knowledge is exactly a reduction to the trivial
    claim.
    """


@dataclass(frozen=True)
class ProveResult(Generic[ReducedClaim, ReductionProof, TranscriptT]):
    """A reduced claim, its conditional reduction proof, and the transcript."""

    reduced_claim: ReducedClaim
    reduction_proof: ReductionProof
    transcript: TranscriptT


@dataclass(frozen=True)
class VerifyResult(Generic[ReducedClaim, TranscriptT]):
    """The verifier-derived reduced claim, advanced transcript, and verdict."""

    reduced_claim: ReducedClaim
    transcript: TranscriptT
    ok: Array


class ProverStage(
    Protocol[Claim_contra, Witness_contra, ReducedClaim, ReductionProof, TranscriptT]
):
    """The prover role of one conditional claim reduction.

    A role that grinds declares `GrindingTranscript` as its transcript
    parameter; one needing only the base seam omits it.
    """

    @abstractmethod
    def prove(
        self,
        claim: Claim_contra,
        witness: Witness_contra,
        transcript: TranscriptT,
    ) -> ProveResult[ReducedClaim, ReductionProof, TranscriptT]: ...


class VerifierStage(
    Protocol[Claim_contra, ReducedClaim, ReductionProof_contra, TranscriptT]
):
    """The verifier role of the same conditional claim reduction."""

    @abstractmethod
    def verify(
        self,
        claim: Claim_contra,
        reduction_proof: ReductionProof_contra,
        transcript: TranscriptT,
    ) -> VerifyResult[ReducedClaim, TranscriptT]: ...
