# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The Spartan composite stage and its R1CS data model.

`Spartan` owns the outer zerocheck, inner lincheck, and witness-opening stages.
Call its paired `prove` and `verify` methods with one `SpartanClaim`; proving
also requires `SpartanWitness`. The object makes their non-linear dataflow explicit.
"""

from zorch.spartan.r1cs import R1CS, assignment
from zorch.spartan.spartan import (
    Spartan,
    SpartanClaim,
    SpartanProof,
    SpartanWitness,
)

__all__ = [
    "R1CS",
    "Spartan",
    "SpartanProof",
    "SpartanClaim",
    "SpartanWitness",
    "assignment",
]
