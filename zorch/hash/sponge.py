"""Padding-free, overwrite-mode sponge hash — scheme-agnostic over a Permutation.

Absorb the input in `rate`-sized blocks, overwriting the first `rate` lanes of
the state (replace, not XOR) and permuting after each; no padding is added, so a
final partial block overwrites only its own lanes. Squeeze the first `out` lanes.
This is the Merkle leaf hasher (Plonky3 PaddingFreeSponge).

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. The block loop
unrolls (input length is static), so the body stays straight-line; it runs one
permute per absorbed block (one for a single-block input), each capturable on the
poseidon2 fusion path (#25).
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

    Contract: rate >= 1 and out >= 1 (validated here); rate < permutation.width
    and out <= permutation.width (validated by ``Sponge``, which knows the width).
    """

    rate: int
    out: int

    def __post_init__(self) -> None:
        if self.rate < 1:
            raise ValueError(f"rate ({self.rate}) must be >= 1")
        if self.out < 1:
            raise ValueError(f"out ({self.out}) must be >= 1")


class Sponge:
    """Padding-free, overwrite-mode sponge over a fixed-width Permutation.

    hash = overwrite state[:rate] with each input block -> permute (repeat) ->
    first `out` lanes. One call is one function — the unit that lowers to one
    fused kernel.
    """

    def __init__(self, permutation: Permutation, params: SpongeParams) -> None:
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
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        state = jnp.zeros(self._permutation.width, dtype=input.dtype)
        for start in range(0, input.shape[0], self.rate):
            block = input[start : start + self.rate]
            state = state.at[: block.shape[0]].set(block)
            state = self._permutation.permute(state)
        return state[: self.out]
