# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The polynomial-commitment seam: `PcsProver` and `PcsVerifier`.

This is the one abstraction every Modern SNARK's PCS plugs into — FRI / Basefold /
WHIR back it with a Merkle commitment plus fold rounds, KZG backs it with an MSM
over a structured reference string. Both satisfy the *same* two protocols; the
only structural difference is that `open` runs many fold rounds for the FRI family
and zero for KZG (its single-point opening is non-interactive — the transcript
only feeds a batching challenge when there is more than one (poly, point)).

**Why two protocols, not one `Pcs`.** `commit`/`open` are the prover's; `verify`
is the verifier's. They are split for the same two reasons the sumcheck block
splits prover/verifier: (1) `open` is an interactive sub-protocol that threads the
Fiat-Shamir transcript, and (2) the prover and verifier hold *asymmetric* keys —
a KZG prover key is O(degree) (the SRS powers) while the verifier key is O(1).
A deployed verifier must never carry the prover's key, so the boundary is a type,
not a convention. A static commitment primitive (the Merkle `Mmcs`) has neither
property and so stays a single unified building block — the split lives in the PCS
layer that *uses* it, not in the primitive.

**Wire types are generic parameters, conformance is mypy-enforced.** The
commitment, retained prover data, and proof are scheme-defined: `PcsProver[C, D,
P]` produces `C` and `D` from `commit` and `P` from `open`; `PcsVerifier[C, P]`
consumes the two wire types. Each instance parameterizes its seam conformance
pin (docs/conventions.md "Seam conformance pins") with them. A pin bites only on
zorch-owned nominal types — which is why every scheme names its prover data
(`KzgProverData`, `FriProverData`, `BasefoldProverData`) instead of passing raw
containers. See docs/pcs.md "Instance anatomy".

**Representation is the scheme's business.** The seam takes polynomials in whatever
form the scheme needs: KZG wants the coefficient basis (powers-of-tau MSM), the
FRI family wants evaluations over a domain. Neither a `PolynomialSpace` nor any
AIR/quotient-commitment index belongs here — those are FRI-implementation or
consumer concerns, kept out so no scheme's shape ossifies into the seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from jax import Array

from zorch.transcript import Transcript

# Scheme-defined wire types. The prover produces the commitment and proof
# (covariant) and threads its retained prover data from `commit` to `open`
# (invariant); the verifier only consumes its two (contravariant).
C_co = TypeVar("C_co", covariant=True)
D = TypeVar("D")
P_co = TypeVar("P_co", covariant=True)
C_contra = TypeVar("C_contra", contravariant=True)
P_contra = TypeVar("P_contra", contravariant=True)


class PcsProver(Protocol[C_co, D, P_co]):
    """Commit to polynomials and prove their evaluations. Holds the (possibly
    O(degree)) prover key."""

    def commit(self, polys: Sequence[Array]) -> tuple[C_co, D]:
        """Bind to a batch of polynomials. Returns the commitment (sent to the
        verifier) and the prover data retained for `open`."""
        ...

    def open(
        self,
        prover_data: D,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, P_co, Transcript]:
        """Prove the evaluations at `points`, threading Fiat-Shamir. Returns
        `(values, proof, transcript)`. The FRI family runs fold rounds here; KZG
        runs none."""
        ...


class PcsVerifier(Protocol[C_contra, P_contra]):
    """Check a claimed opening against a commitment. Holds only the O(1)
    verifier key."""

    def verify(
        self,
        commitment: C_contra,
        points: Sequence[Array],
        values: Array,
        proof: P_contra,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Return `(ok, transcript)` where `ok` is a scalar boolean array."""
        ...
