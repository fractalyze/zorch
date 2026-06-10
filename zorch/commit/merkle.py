"""Layer-by-layer k-ary Merkle commitment — scheme-agnostic, on Sponge + Compression.

`commit` hashes each matrix row to a leaf digest (Sponge), then folds sibling
groups per layer (Compression, whose `arity` sets the tree's) down to a single
root, returning `(raw_root, digest_layers)` (leaf digests first, root last).
It adds NO domain separator — that, the proof layout, and the verify error codes
are scheme-specific and live in the consumer (e.g. whir-zorch's SMCS).

A level whose node count is not a multiple of the arity is completed with
zero digests — the only convention that keeps a k-ary tree well-defined on
power-of-two heights (e.g. 2^k leaves under arity 4 leave a 2-node top level),
and the one the k-ary schemes this seam serves use. The padded form is what
`digest_layers` stores, so an opening near the boundary can read its zero
siblings like any others. A binary tree on a power-of-two height never pads,
so the arity-2 layout is exactly the historical one.

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
    path: list[Array]  # each (digest_elems,) for arity 2, (arity-1, digest_elems) above


class MerkleTree:
    """A k-ary Merkle commitment over a single matrix.

    `leaf_hasher` squeezes each row to a `digest_elems`-element leaf; `compressor`
    folds `compressor.arity` digests into one — the tree's arity follows it.
    They must agree on digest size (`leaf_hasher.out == compressor.chunk`).

    Arity 2 keeps the historical path layout (each path entry one sibling,
    shape `(digest_elems,)`); a wider arity carries the whole sibling group per
    level, shape `(arity-1, digest_elems)`. The batched `reconstruct_roots`
    fast path stays binary-only — its sole consumer is the binary fold-PCS
    query machinery.
    """

    def __init__(self, leaf_hasher: Sponge, compressor: Compression) -> None:
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self.arity = compressor.arity
        self.digest_elems = compressor.chunk
        # Wrap the whole commit in the agnostic composite only when both blocks
        # lower to a hash-dedicated marker the vendor can expand; otherwise the
        # marker would be unexpandable, so fall back to the plain vmap path.
        self._fused = (
            leaf_hasher.has_dedicated_fusion and compressor.has_dedicated_fusion
        )

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a (height, width) matrix.

        Returns `(raw_root (digest_elems,), digest_layers)`, where digest_layers
        runs leaf digests -> ... -> root, each (nodes_at_level, digest_elems)
        in the zero-padded form (see the module docstring).

        A binary tree keeps its historical power-of-two-height contract — its
        pad-free layout is what the fold-PCS query machinery indexes; k-ary
        consumers commit any height via the per-level padding.
        """
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
        if self.arity == 2 and not is_power_of_two(matrix.shape[0]):
            raise ValueError(
                f"matrix height ({matrix.shape[0]}) must be a power of two"
            )
        # The validation above guards eagerly, so it stays outside the marked
        # region; only `matrix` is an explicit operand (constants auto-lift).
        if self._fused:
            return fused_region(self._build, matrix, name=MERKLE_COMMIT_MARKER)
        return self._build(matrix)

    def _build(self, matrix: Array) -> tuple[Array, list[Array]]:
        """The traced commit body: vmap the leaf hash, fold arity-sized groups
        per layer down to the root. Wrapped by `commit` as one
        `zorch.merkle_commit` region."""
        layer = jax.vmap(self._leaf_hasher.hash)(matrix)
        digest_layers = [layer]
        while layer.shape[0] > 1:
            rem = layer.shape[0] % self.arity
            if rem:
                # Complete the level with zero digests and store the padded
                # form: an opening adjacent to the boundary reads its zero
                # siblings from the layer like any others.
                pad = jnp.zeros((self.arity - rem, self.digest_elems), layer.dtype)
                layer = jnp.concatenate([layer, pad])
                digest_layers[-1] = layer
            groups = layer.reshape(-1, self.arity, self.digest_elems)
            layer = jax.vmap(self._compressor.compress)(groups)
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
            if self.arity == 2:
                path.append(digest_layers[level][idx ^ 1])
            else:
                # The whole sibling group except the node itself, in level
                # order: slot j of the entry is group position j + (j >= pos).
                pos = idx % self.arity
                group_start = (idx // self.arity) * self.arity
                j = jnp.arange(self.arity - 1)
                path.append(digest_layers[level][group_start + j + (j >= pos)])
            idx //= self.arity
        return Opening(row=matrix[index], path=path)

    def _fold_with_sibling(
        self, node: Array, idx: Array, sibling: Array
    ) -> tuple[Array, Array]:
        """Compress `node` with its `sibling` into the parent digest, ordering
        the pair by the running leaf-index parity `idx` (data-select, not a
        Python branch, so the fold traces under `vmap`); return the parent and
        `idx` halved for the level above."""
        is_left = idx % 2 == 0
        left = jnp.where(is_left, node, sibling)
        right = jnp.where(is_left, sibling, node)
        return self._compressor.compress(jnp.stack([left, right])), idx // 2

    def _fold_with_siblings(
        self, node: Array, idx: Array, siblings: Array
    ) -> tuple[Array, Array]:
        """k-ary dual of `_fold_with_sibling`: re-insert `node` at its group
        position among `siblings` `(arity-1, digest_elems)` by data-select (no
        Python branch, so the fold traces under `vmap`), compress the group,
        and return the parent with `idx` divided for the level above."""
        pos = idx % self.arity
        i = jnp.arange(self.arity)
        # Slot i holds the sibling that `open` packed for it (skipping the
        # node's own slot), except slot pos, which holds the node.
        gathered = siblings[jnp.clip(i - (i > pos), 0, self.arity - 2)]
        group = jnp.where((i == pos)[:, None], node[None, :], gathered)
        return self._compressor.compress(group), idx // self.arity

    def reconstruct_root(self, index: int | Array, opening: Opening) -> Array:
        """Rebuild the raw root from an `opening`'s row + path (leaf-first).

        Returns the root Array, not a verdict — a separator-binding consumer
        (e.g. SP1's SMCS) rebinds the raw root before comparing, which
        `verify`'s plain equality can't express. Single-index; batch by
        `jax.vmap`-ing over `(index, opening)`."""
        node = self._leaf_hasher.hash(opening.row)
        if not opening.path:
            return node

        # Roll the leaf->root fold into one `scan`: the body (one compress) is
        # traced once, so a depth-h path lowers to a single compress region
        # rather than h unrolled copies. The unrolled fold dominated the
        # verifier's trace+lower across its per-layer reconstruct chains (#163).
        def fold(
            carry: tuple[Array, Array], siblings: Array
        ) -> tuple[tuple[Array, Array], None]:
            node, idx = carry
            if self.arity == 2:
                return self._fold_with_sibling(node, idx, siblings), None
            return self._fold_with_siblings(node, idx, siblings), None

        (node, _), _ = jax.lax.scan(fold, (node, index), jnp.stack(opening.path))
        return node

    def reconstruct_roots(
        self, rows: Array, indices: Array, paths: Array, valid: Array
    ) -> Array:
        """Rebuild a whole batch of roots in one `vmap` + one `scan`.

        `rows` `(B, row_width)`, `indices` `(B,)`, `paths` `(B, depth, digest)`
        leaf-first, `valid` `(B, depth)` true for the real levels of each path
        and false for trailing don't-care padding. Paths shorter than `depth`
        are zero-padded and masked, so a batch can mix Merkle trees of different
        height: a masked step keeps the running node, so element `b` rebuilds the
        same root `reconstruct_root` would on its first `valid[b].sum()` levels.

        Tracing the compress body once for the whole batch — instead of once per
        `reconstruct_root` call — is the point: the folding verifiers reconstruct
        one pair per fold layer, and that per-layer loop dominated their
        trace+lower (#163)."""
        if self.arity != 2:
            raise NotImplementedError(
                "reconstruct_roots is binary-only: its consumer is the binary "
                "fold-PCS query machinery; use vmap(reconstruct_root) for k-ary"
            )

        def one(row: Array, index: Array, path: Array, mask: Array) -> Array:
            node = self._leaf_hasher.hash(row)

            def fold(
                carry: tuple[Array, Array], step: tuple[Array, Array]
            ) -> tuple[tuple[Array, Array], None]:
                node, idx = carry
                sibling, active = step
                folded, idx = self._fold_with_sibling(node, idx, sibling)
                # Padding past the real depth is a no-op: keep the rebuilt root.
                node = jnp.where(active, folded, node)
                return (node, idx), None

            (node, _), _ = jax.lax.scan(fold, (node, index), (path, mask))
            return node

        return jax.vmap(one)(rows, indices, paths, valid)

    def verify(self, root: Array, index: int, opening: Opening) -> bool:
        """Rebuild the root from the row + path; compare to the committed root."""
        if not 0 <= index < self.arity ** len(opening.path):
            return False
        return bool(jnp.array_equal(self.reconstruct_root(index, opening), root))
