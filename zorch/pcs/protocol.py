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

**Representation is the scheme's business.** The seam takes polynomials in whatever
form the scheme needs: KZG wants the coefficient basis (powers-of-tau MSM), the
FRI family wants evaluations over a domain. Neither a `PolynomialSpace` nor any
AIR/quotient-commitment index belongs here — those are FRI-implementation or
consumer concerns, kept out so no scheme's shape ossifies into the seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from jax import Array

from zorch.transcript import Transcript


class PcsProver(Protocol):
    """Commit to polynomials and prove their evaluations. Holds the (possibly
    O(degree)) prover key. The commitment, prover data, and proof are opaque,
    scheme-defined types (`Any` here); each instance pins them concretely."""

    def commit(self, polys: Sequence[Array]) -> tuple[Any, Any]:
        """Bind to a batch of polynomials. Returns the commitment (sent to the
        verifier) and opaque prover data kept for `open`."""
        ...

    def open(
        self,
        prover_data: Any,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, Any, Transcript]:
        """Prove the evaluations at `points`, threading Fiat-Shamir. Returns
        `(values, proof, transcript)`. The FRI family runs fold rounds here; KZG
        runs none."""
        ...


class PcsVerifier(Protocol):
    """Check a claimed opening against a commitment. Holds only the O(1)
    verifier key."""

    def verify(
        self,
        commitment: Any,
        points: Sequence[Array],
        values: Array,
        proof: Any,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Return `(ok, transcript)` where `ok` is a scalar boolean array."""
        ...
