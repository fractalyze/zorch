# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generic Spartan R1CS combinators, assembled from zorch blocks.

Field- and PCS-agnostic R1CS-proving combinators — a zerocheck (outer sumcheck),
a random-linear-combination batcher, a lincheck (inner sumcheck), and a
sumcheck-claim→PCS-opening glue — plus a thin `prove` / `verify` assembly wiring
them into the Spartan PIOP. The combinators carry no scheme assumption; the
assembly is the reference schedule. See `docs/spartan.md`.
"""

from zorch.spartan.r1cs import R1CS, assignment
from zorch.spartan.spartan import SpartanProof, prove, verify

__all__ = ["R1CS", "assignment", "SpartanProof", "prove", "verify"]
