"""The Permutation seam every symmetric primitive builds on.

A fixed-width permutation over a single field dtype. Consumers (duplex sponge,
Fiat-Shamir transcript, Merkle compression) read `width` to size state and
`dtype` to allocate it, then call `permute` — they never name a concrete hash.
Poseidon2 is one implementation; any other fixed-width permutation drops in
unchanged.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Permutation(Protocol):
    width: int          # state length (rate + capacity)
    dtype: Any          # field dtype of each state element

    def permute(self, state: Array) -> Array:
        """Apply the permutation: (width,) over `dtype` -> (width,).

        One call is one function — the unit that lowers to one fused kernel.
        """
        ...
