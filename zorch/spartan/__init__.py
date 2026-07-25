# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable Spartan roles and their shared R1CS data model."""

from zorch.spartan.r1cs import R1CS, assignment
from zorch.spartan.spartan import (
    SpartanClaim,
    SpartanProof,
    SpartanProver,
    SpartanVerifier,
    SpartanWitness,
)

__all__ = [
    "R1CS",
    "SpartanClaim",
    "SpartanProof",
    "SpartanProver",
    "SpartanVerifier",
    "SpartanWitness",
    "assignment",
]
