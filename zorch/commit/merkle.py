"""Layer-by-layer Merkle commitment — scheme-agnostic, built on Sponge + Compression.

`commit` hashes each matrix row to a leaf digest (Sponge), then folds `arity`
children into one parent per layer (Compression) until a single root remains. It
returns the raw root and the digest layers (leaf digests first, root last) and
adds NO domain separator — that, the proof layout, and the verify error codes are
scheme-specific and live in the consumer (e.g. whir-zorch's SMCS).

Each layer is one `vmap` over its nodes, so it lowers to one fused kernel per
layer (cuda-graph-friendly); the layer count is static, so the fold unrolls.
"""

from __future__ import annotations

import jax
from jax import Array

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge


def _is_power_of(n: int, base: int) -> bool:
    if n < 1:
        return False
    while n % base == 0:
        n //= base
    return n == 1


class MerkleTree:
    """A `arity`-ary Merkle commitment over a single matrix.

    `leaf_hasher` squeezes each row to a `digest_elems`-element leaf; `compressor`
    folds `arity` digests into one. The two must agree on digest size
    (leaf_hasher.out == compressor.chunk).
    """

    def __init__(self, leaf_hasher: Sponge, compressor: Compression):
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self._arity = compressor.arity
        self.digest_elems = compressor.chunk

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a (height, width) matrix, height a power of `arity`.

        Returns (raw_root (digest_elems,), digest_layers), where digest_layers
        runs leaf digests -> ... -> root, each (nodes_at_level, digest_elems).
        """
        height = matrix.shape[0]
        if not _is_power_of(height, self._arity):
            raise ValueError(
                f"matrix height ({height}) must be a power of arity " f"({self._arity})"
            )
        layer = jax.vmap(self._leaf_hasher.hash)(matrix)
        digest_layers = [layer]
        while layer.shape[0] > 1:
            groups = layer.reshape(-1, self._arity, self.digest_elems)
            layer = jax.vmap(self._compressor.compress)(groups)
            digest_layers.append(layer)
        return digest_layers[-1][0], digest_layers
