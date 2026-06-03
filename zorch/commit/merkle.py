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

When both blocks lower to a hash-dedicated fusion marker (`has_dedicated_fusion`),
`commit` wraps the whole tree in one `zorch.merkle_commit` composite — a
hash-agnostic, whole-tree boundary a vendor expands into the per-layer hash
kernels (the nested permute markers carry the hash identity). The wrapping passes
only `matrix`; the round constants ride in as auto-lifted composite operands, so
this stays scheme-agnostic (no permutation/hash is named here). Otherwise (or on a
jaxlib without `stablehlo.CompositeOp`) the plain vmap path runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.fusion import fused_region
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import is_power_of_two

# Hash-agnostic whole-tree boundary: `commit` wraps the entire tree in one
# composite under this name (never a per-hash name). A vendor expander reads the
# hash identity from the nested `permute` marker, so adding a hash never touches
# this marker or MerkleTree — see the module docstring.
MERKLE_COMMIT_MARKER = "zorch.merkle_commit"


@partial(jax.tree_util.register_dataclass, data_fields=["row", "path"], meta_fields=[])
@dataclass(frozen=True)
class Opening:
    """A single leaf's authentication path: the committed matrix `row` plus the
    sibling digest at each level (leaf-first, excluding the root).

    A pytree (leaves: `row` and each path sibling) so `open` / `reconstruct_root`
    batch under `jax.vmap` and trace under `jit`."""

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
        # Wrap the whole commit in the agnostic composite only when both blocks
        # lower to a hash-dedicated marker the vendor can expand; otherwise the
        # marker would be unexpandable, so fall back to the plain vmap path.
        self._fused = (
            leaf_hasher.has_dedicated_fusion and compressor.has_dedicated_fusion
        )

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
        # The validation above guards eagerly, so it stays outside the marked
        # region; only `matrix` is an explicit operand (constants auto-lift).
        if self._fused:
            return fused_region(self._build, matrix, name=MERKLE_COMMIT_MARKER)
        return self._build(matrix)

    def _build(self, matrix: Array) -> tuple[Array, list[Array]]:
        """The traced commit body: vmap the leaf hash, fold pairs per layer down
        to the root. Wrapped by `commit` as one `zorch.merkle_commit` region."""
        layer = jax.vmap(self._leaf_hasher.hash)(matrix)
        digest_layers = [layer]
        while layer.shape[0] > 1:
            pairs = layer.reshape(-1, 2, self.digest_elems)
            layer = jax.vmap(self._compressor.compress)(pairs)
            digest_layers.append(layer)
        return digest_layers[-1][0], digest_layers

    def open(
        self, matrix: Array, digest_layers: list[Array], index: int | Array
    ) -> Opening:
        """Authentication path for leaf `index`: its row plus each level's sibling.

        Single-index by construction; batch by `jax.vmap`-ing over `index` (the
        sibling gather is orchestration outside the fused permute, like `commit`).
        Index validity is a prover-side precondition — enforced eagerly for any
        concrete index (Python int or 0-d Array), skipped only under tracing,
        where the value is unknown and JAX would silently clamp an out-of-range
        gather; there `verify` owns out-of-range rejection.
        """
        if not isinstance(index, jax.core.Tracer) and not 0 <= index < matrix.shape[0]:
            raise IndexError(f"leaf index {index} out of range [0, {matrix.shape[0]})")
        path = []
        idx = index
        for level in range(len(digest_layers) - 1):  # leaf layer up to below root
            path.append(digest_layers[level][idx ^ 1])
            idx //= 2
        return Opening(row=matrix[index], path=path)

    def reconstruct_root(self, index: int | Array, opening: Opening) -> Array:
        """Rebuild the raw root from an `opening`'s row + path (leaf-first).

        Returns the root Array, not a verdict — a separator-binding consumer
        (e.g. SP1's SMCS) rebinds the raw root before comparing, which
        `verify`'s plain equality can't express. Single-index; batch by
        `jax.vmap`-ing over `(index, opening)`."""
        node = self._leaf_hasher.hash(opening.row)
        idx = index
        for sibling in opening.path:
            # Data-select the sibling order (not a Python branch on idx) so the
            # fold traces under vmap; idx is the running parity at this level.
            is_left = idx % 2 == 0
            left = jnp.where(is_left, node, sibling)
            right = jnp.where(is_left, sibling, node)
            node = self._compressor.compress(jnp.stack([left, right]))
            idx //= 2
        return node

    def verify(self, root: Array, index: int, opening: Opening) -> bool:
        """Rebuild the root from the row + path; compare to the committed root."""
        if not 0 <= index < (1 << len(opening.path)):
            return False
        return bool(jnp.array_equal(self.reconstruct_root(index, opening), root))
