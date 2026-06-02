"""Padding-free, overwrite-mode sponge hash — scheme-agnostic over a Permutation.

Absorb the input in `rate`-sized blocks, overwriting the first `rate` lanes of
the state (replace, not XOR) and permuting after each; no padding is added, so a
final partial block overwrites only its own lanes. Squeeze the first `out` lanes.
This is the Merkle leaf hasher (Plonky3 PaddingFreeSponge).

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. The block loop
unrolls (input length is static), so the body stays straight-line — only the
permutation carries a loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.hash.permutation import Permutation


@dataclass(frozen=True)
class SpongeParams:
    """Free parameters of a padding-free sponge.

    rate : field elements absorbed per permutation (capacity = width - rate).
    out  : field elements squeezed (the digest size).

    Contract (validated by ``Sponge``): rate < permutation.width and
    out <= permutation.width.
    """

    rate: int
    out: int


class Sponge:
    """Padding-free, overwrite-mode sponge over a fixed-width Permutation.

    hash = overwrite state[:rate] with each input block -> permute (repeat) ->
    first `out` lanes. One call is one function — the unit that lowers to one
    fused kernel.
    """

    def __init__(self, permutation: Permutation, params: SpongeParams):
        if params.rate >= permutation.width:
            raise ValueError(
                f"rate ({params.rate}) must be < permutation width "
                f"({permutation.width})"
            )
        if params.out > permutation.width:
            raise ValueError(
                f"out ({params.out}) must be <= permutation width "
                f"({permutation.width})"
            )
        self._permutation = permutation
        self.rate = params.rate
        self.out = params.out

    def hash(self, input: Array) -> Array:
        """Absorb `input` (1-D) and squeeze: (n,) over dtype -> (out,)."""
        state = jnp.zeros(self._permutation.width, dtype=input.dtype)
        for start in range(0, input.shape[0], self.rate):
            block = input[start : start + self.rate]
            state = state.at[: block.shape[0]].set(block)
            state = self._permutation.permute(state)
        return state[: self.out]
