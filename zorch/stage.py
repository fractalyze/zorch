# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Paired proof stages and their typed results.

A stage is a reusable protocol component with paired prover and verifier
behavior. It may drive round recurrences, perform PCS operations, or own child
stages. A composite stage writes its protocol dataflow explicitly with ordinary
Python, preserving fan-out and skip-level dependencies as named values.
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


class Stage(
    ABC,
    Generic[Claim, Witness, ReducedClaim, ReductionProof],
):
    """A paired proof reduction.

    The reduction proof establishes the source claim conditional on the reduced
    claim. It does not normally establish the reduced claim itself.
    """

    @abstractmethod
    def prove(
        self, claim: Claim, witness: Witness, transcript: Transcript
    ) -> ProveResult[ReducedClaim, ReductionProof]: ...

    @abstractmethod
    def verify(
        self,
        claim: Claim,
        reduction_proof: ReductionProof,
        transcript: Transcript,
    ) -> VerifyResult[ReducedClaim]: ...
