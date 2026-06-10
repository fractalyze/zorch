"""Truncated-permutation compression — scheme-agnostic n-to-1 over a Permutation.

Zero-pad `arity` chunks of `chunk` field elements into the permutation width,
permute, take the first `chunk` lanes (collision-resistant in the hash-tree
setting). Width comes from `permutation.width`; `arity` and `chunk` are the only
free parameters, carried on `CompressionParams` like `Poseidon2Params`. Names no
field/scheme/zkVM.

`compress` is one function over the permutation, so a `vmap` over a Merkle layer
batches into one permute; that collapses to one GPU kernel once the permutation
is captured to a kernel (the poseidon2 fusion path, #25).
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

    Contract: arity >= 2 and chunk >= 1 (validated here); arity * chunk <=
    permutation.width (validated by ``Compression``, which knows the width).
    """

    arity: int
    chunk: int

    def __post_init__(self) -> None:
        if self.arity < 2:
            raise ValueError(f"arity ({self.arity}) must be >= 2")
        if self.chunk < 1:
            raise ValueError(f"chunk ({self.chunk}) must be >= 1")


class Compression:
    """n-to-1 truncated-permutation compression over a fixed-width Permutation.

    compress = place arity*chunk input lanes into a zero pre-image of width
    `permutation.width` -> permute -> first `chunk` lanes. One call is one
    function — the unit that lowers to one fused kernel.
    """

    def __init__(self, permutation: Permutation, params: CompressionParams) -> None:
        used = params.arity * params.chunk
        if used > permutation.width:
            raise ValueError(
                f"arity*chunk ({used}) must be <= permutation width "
                f"({permutation.width})"
            )
        self._permutation = permutation
        self.arity = params.arity
        self.chunk = params.chunk

    # Value equality/hash for static jit-zone keys, like the Permutation
    # contract it builds on (#214) — identity equality re-traces per instance.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Compression):
            return NotImplemented
        return (self._permutation, self.arity, self.chunk) == (
            other._permutation,
            other.arity,
            other.chunk,
        )

    def __hash__(self) -> int:
        return hash((self._permutation, self.arity, self.chunk))

    @property
    def has_dedicated_fusion(self) -> bool:
        """Whether the permutation lowers to a hash-dedicated fusion marker, so a
        consumer can wrap a whole region using this hash (e.g. a Merkle commit) in
        an expandable composite. Delegates to the permutation; names no hash."""
        return self._permutation.has_dedicated_fusion

    def compress(self, inputs: Array) -> Array:
        """Compress `arity` chunks into one: (arity, chunk) over dtype -> (chunk,)."""
        if inputs.shape != (self.arity, self.chunk):
            raise ValueError(
                f"inputs shape must be {(self.arity, self.chunk)}, got {inputs.shape}"
            )
        pre = jnp.zeros(self._permutation.width, dtype=inputs.dtype)
        pre = pre.at[: self.arity * self.chunk].set(inputs.reshape(-1))
        return self._permutation.permute(pre)[: self.chunk]
