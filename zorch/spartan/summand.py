# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The zerocheck summand `eq·(â◦b̂ − ĉ)` (`eq = eq(τ,·)`; `â,b̂,ĉ = Az,Bz,Cz`,
the committed products, `◦` Hadamard — paper §3) for the outer sumcheck.

Satisfies the `zorch.sumcheck.prover.SumcheckSummand` seam, so `StandardRound`
drives it like the product summand — only `combine` differs. The combine is
mixed-degree (`eq·(â◦b̂)` deg 3, `eq·ĉ` deg 2), so it must sample on the finite
`natural_domain(3)` (a leading-∞ domain needs a homogeneous summand); that is
`StandardRound`'s default when `domain` is `None`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from frx import Array


@dataclass(frozen=True)
class ZerocheckSummand:
    """The outer-sumcheck summand `eq·(â◦b̂ − ĉ)` (degree 3), read by the round-poly
    builder through the `SumcheckSummand` seam."""

    @property
    def degree(self) -> int:
        return 3

    def combine_scalars(self) -> tuple[Array, ...]:
        """No loop-invariant scalars: the summand reads only its four factors."""
        return ()

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        """`eq·(â◦b̂ − ĉ)` — the scalar-explicit seam form (no scalars here). Single
        source of the combine math; `_combine` routes here so they cannot drift."""
        del scalars
        eq, a, b, c = factors
        return eq * (a * b - c)

    def _combine(self, *factors: Array) -> Array:
        """Bound to its (empty) scalars; the round-poly builder reads only this."""
        return self.combine(self.combine_scalars(), *factors)
