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

ProverInput = TypeVar("ProverInput")
ProverOutput = TypeVar("ProverOutput")
VerifierInput = TypeVar("VerifierInput")
VerifierOutput = TypeVar("VerifierOutput")
Proof = TypeVar("Proof")


@dataclass(frozen=True)
class ProveResult(Generic[ProverOutput, Proof]):
    """One stage's typed output, proof section, and advanced transcript."""

    output: ProverOutput
    proof: Proof
    transcript: Transcript


@dataclass(frozen=True)
class VerifyResult(Generic[VerifierOutput]):
    """One stage dual's typed output, advanced transcript, and verdict."""

    output: VerifierOutput
    transcript: Transcript
    ok: Array


class Stage(
    ABC,
    Generic[ProverInput, ProverOutput, VerifierInput, VerifierOutput, Proof],
):
    """A paired prover/verifier protocol component.

    Each method accepts one semantic domain value. Prover and verifier inputs
    and outputs may differ because they carry different knowledge. Use `None`
    when a side needs no input beyond its proof and transcript.
    """

    name: str

    @abstractmethod
    def prove(
        self, inputs: ProverInput, transcript: Transcript
    ) -> ProveResult[ProverOutput, Proof]: ...

    @abstractmethod
    def verify(
        self, inputs: VerifierInput, proof: Proof, transcript: Transcript
    ) -> VerifyResult[VerifierOutput]: ...
