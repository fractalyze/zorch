# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The zerocheck relation `â◦b̂ − ĉ` (`â,b̂,ĉ = Az,Bz,Cz`, `◦` Hadamard).

The equality-polynomial engine supplies `eq(τ,·)` as a factored linear weight,
so it is deliberately absent here. The combine is mixed-degree (`â◦b̂` has
degree 2 and `ĉ` degree 1), which requires a finite sampling domain rather than
an infinity-leading homogeneous-product domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from frx import Array


@dataclass(frozen=True)
class ZerocheckSummand:
    """The three-factor zerocheck relation `â◦b̂ − ĉ` (degree 2)."""

    @property
    def degree(self) -> int:
        return 2

    def combine_scalars(self) -> tuple[Array, ...]:
        """No loop-invariant scalars: the summand reads only its three factors."""
        return ()

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        """`â◦b̂ − ĉ` — the scalar-explicit seam form (no scalars here)."""
        del scalars
        a, b, c = factors
        return a * b - c

    def _combine(self, *factors: Array) -> Array:
        """Bound to its (empty) scalars; the round-poly builder reads only this."""
        return self.combine(self.combine_scalars(), *factors)
