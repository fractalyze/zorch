# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The polynomial-commitment seam as a committer plus a terminal stage.

A PCS is two things wearing one name: `commit` runs before any claim exists, so
it reduces nothing, while opening is a claim reduction in everything but name.

Keep the halves apart. Merging them would put a KZG prover key — O(degree),
against an O(1) verifier key — within reach of a deployed verifier.
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
    """Bind to a batch of polynomials, creating the object later claims are
    about. Holds the proving material, so a verifier never constructs one."""

    def commit(self, polys: Sequence[Array]) -> tuple[C_co, D_co]:
        """Return the commitment sent to the verifier and the prover data
        retained for the opening."""
        ...


@dataclass(frozen=True)
class OpeningClaim(Generic[Commitment]):
    """The committed polynomials evaluate at `points`.

    No values: the prover computes them while opening, so a claim naming them
    would be one neither role could construct before the fact.
    """

    commitment: Commitment
    points: Sequence[Array]


@dataclass(frozen=True)
class OpeningWitness(Generic[ProverData]):
    """The prover data `commit` retained; prover-only, hence not in the claim."""

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
