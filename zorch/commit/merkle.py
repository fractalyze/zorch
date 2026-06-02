"""Layer-by-layer binary Merkle commitment — scheme-agnostic, on Sponge + Compression.

`commit` hashes each matrix row to a leaf digest (Sponge), then folds sibling
pairs per layer (Compression) down to a single root, returning
`(raw_root, digest_layers)` (leaf digests first, root last). It adds NO domain
separator — that, the proof layout, and the verify error codes are scheme-specific
and live in the consumer (e.g. whir-zorch's SMCS).

Each layer is one `vmap` over its nodes, and the fold unrolls (layer count is
static), so no host-driven loop appears. An internal layer batches one
`compress` = one permute; the leaf layer batches one `hash` = one permute per
absorbed block. Those collapse to one GPU kernel per permute once the
permutation itself is captured to a kernel (the poseidon2 fusion path, #25).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import is_power_of_two


@dataclass(frozen=True)
class Opening:
    """A single leaf's authentication path: the committed matrix `row` plus the
    sibling digest at each level (leaf-first, excluding the root)."""

    row: Array
    path: list[Array]  # each (digest_elems,)


class MerkleTree:
    """A binary Merkle commitment over a single matrix.

    `leaf_hasher` squeezes each row to a `digest_elems`-element leaf; `compressor`
    folds two digests into one. They must agree on digest size
    (`leaf_hasher.out == compressor.chunk`), and the compressor must be 2-to-1.
    """

    def __init__(self, leaf_hasher: Sponge, compressor: Compression) -> None:
        if compressor.arity != 2:
            raise ValueError(
                f"MerkleTree builds a binary tree; compressor arity must be 2, "
                f"got {compressor.arity}"
            )
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self.digest_elems = compressor.chunk

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a (height, width) matrix, height a power of two.

        Returns `(raw_root (digest_elems,), digest_layers)`, where digest_layers
        runs leaf digests -> ... -> root, each (nodes_at_level, digest_elems).
        """
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
        if not is_power_of_two(matrix.shape[0]):
            raise ValueError(
                f"matrix height ({matrix.shape[0]}) must be a power of two"
            )
        layer = jax.vmap(self._leaf_hasher.hash)(matrix)
        digest_layers = [layer]
        while layer.shape[0] > 1:
            pairs = layer.reshape(-1, 2, self.digest_elems)
            layer = jax.vmap(self._compressor.compress)(pairs)
            digest_layers.append(layer)
        return digest_layers[-1][0], digest_layers

    def open(self, matrix: Array, digest_layers: list[Array], index: int) -> Opening:
        """Authentication path for leaf `index`: its row plus each level's sibling."""
        if not 0 <= index < matrix.shape[0]:
            raise IndexError(f"leaf index {index} out of range [0, {matrix.shape[0]})")
        path = []
        idx = index
        for level in range(len(digest_layers) - 1):  # leaf layer up to below root
            path.append(digest_layers[level][idx ^ 1])
            idx //= 2
        return Opening(row=matrix[index], path=path)

    def verify(self, root: Array, index: int, opening: Opening) -> bool:
        """Rebuild the root from the row + path; compare to the committed root."""
        if not 0 <= index < (1 << len(opening.path)):
            return False
        node = self._leaf_hasher.hash(opening.row)
        idx = index
        for sibling in opening.path:
            pair = (
                jnp.stack([node, sibling])
                if idx % 2 == 0
                else jnp.stack([sibling, node])
            )
            node = self._compressor.compress(pair)
            idx //= 2
        return bool(jnp.array_equal(node, root))
