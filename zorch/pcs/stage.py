# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The polynomial-commitment seam as a committer plus a terminal stage.

A PCS is two things wearing one name. `commit` runs *before any claim exists* —
it creates the object later claims are about — so it is a committer, not a
reduction. Opening is a reduction in everything but name: the claim is "the
polynomials behind this commitment evaluate at these points", the witness is the
retained prover data, and it reduces to `TrivialClaim` because an opening closes
its claim rather than passing one on.

That split retires the separate `PcsProver`/`PcsVerifier` pair. Its stated reason
was key asymmetry — a KZG prover key is O(degree) while the verifier key is O(1),
so the roles must be separately constructible — and `ProverStage`/`VerifierStage`
already encode exactly that isolation. What genuinely remained different was only
`commit`, which is what `Committer` now names.

The evaluations travel in the *proof*, not the claim: the prover computes them
while opening, and the verifier learns them from the wire and checks them against
the commitment. A claim carrying values the prover has not produced yet would be
a claim neither role could construct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from frx import Array

Commitment = TypeVar("Commitment")
ProverData = TypeVar("ProverData")
Proof = TypeVar("Proof")

C_co = TypeVar("C_co", covariant=True)
D_co = TypeVar("D_co", covariant=True)
P_co = TypeVar("P_co", covariant=True)


class Committer(Protocol[C_co, D_co]):
    """Bind to a batch of polynomials.

    Not a stage: no claim exists yet to reduce. Holds the (possibly O(degree))
    proving material, so a deployed verifier never constructs one.
    """

    def commit(self, polys: Sequence[Array]) -> tuple[C_co, D_co]:
        """Return the commitment sent to the verifier and the prover data
        retained for the opening."""
        ...


@dataclass(frozen=True)
class OpeningClaim(Generic[Commitment]):
    """The committed polynomials evaluate at `points`.

    Both roles hold this before the opening runs; the evaluations themselves are
    the prover's to supply and arrive in `OpeningProof`.
    """

    commitment: Commitment
    points: Sequence[Array]


@dataclass(frozen=True)
class OpeningWitness(Generic[ProverData]):
    """The prover data `commit` retained — prover-only, so it never appears in
    the claim."""

    prover_data: ProverData


class CommittingOpener(Protocol[Commitment, ProverData, P_co]):
    """A committer that also opens — what a consumer holding both halves needs.

    Structural, because "commits and opens" is a conjunction of two independent
    contracts and Python cannot spell their intersection nominally. A scheme
    still subclasses `ProverStage` for the opening itself; this only names the
    pair for call sites like Spartan that commit and later open.
    """

    def commit(self, polys: Sequence[Array]) -> tuple[Commitment, ProverData]: ...

    def prove(
        self,
        claim: OpeningClaim[Commitment],
        witness: OpeningWitness[ProverData],
        transcript: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class OpeningProof(Generic[Proof]):
    """The claimed evaluations and the scheme's opening proof for them."""

    values: Array
    proof: Proof
