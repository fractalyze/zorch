"""Truncated-permutation compression — scheme-agnostic n-to-1 over a Permutation.

Zero-pad `arity` chunks of `chunk` field elements into the permutation width,
permute, take the first `chunk` lanes (collision-resistant in the hash-tree
setting). Width comes from `permutation.width`; `arity` and `chunk` are the only
free parameters, carried on `CompressionParams` like `Poseidon2Params`. Names no
field/scheme/zkVM.

`compress` is one function over the permutation, so it inherits the permutation's
single fused kernel; a `vmap` over a Merkle layer keeps one kernel per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.hash.permutation import Permutation


@dataclass(frozen=True)
class CompressionParams:
    """Free parameters of a truncated-permutation compression.

    arity : number of input chunks (the n in n-to-1).
    chunk : field elements per chunk == digest size == output length.

    Contract (validated by ``Compression``): arity * chunk <= permutation.width,
    so the padded pre-image fits the state.
    """

    arity: int
    chunk: int


class Compression:
    """n-to-1 truncated-permutation compression over a fixed-width Permutation.

    compress = place arity*chunk input lanes into a zero pre-image of width
    `permutation.width` -> permute -> first `chunk` lanes. One call is one
    function — the unit that lowers to one fused kernel.
    """

    def __init__(self, permutation: Permutation, params: CompressionParams):
        used = params.arity * params.chunk
        if used > permutation.width:
            raise ValueError(
                f"arity*chunk ({used}) must be <= permutation width "
                f"({permutation.width})"
            )
        self._permutation = permutation
        self.arity = params.arity
        self.chunk = params.chunk

    def compress(self, inputs: Array) -> Array:
        """Compress `arity` chunks into one: (arity, chunk) over dtype -> (chunk,)."""
        pre = jnp.zeros(self._permutation.width, dtype=inputs.dtype)
        pre = pre.at[: self.arity * self.chunk].set(inputs.reshape(-1))
        return self._permutation.permute(pre)[: self.chunk]
