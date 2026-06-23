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

Each layer is one `vmap` over its nodes: an internal layer batches one
`compress` = one permute; the leaf layer batches one `hash` = one permute per
absorbed block. Those collapse to one GPU kernel per permute once the
permutation itself is captured to a kernel (the poseidon2 fusion path, #25). A
binary tree folds the layers with one `lax.scan` so the traced body is O(1) in
tree height — the unrolled fold dominated the trace-commit compile; a k-ary
tree keeps the unrolled fold (see `_build` for why).

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


def _regular_fold_height(leaf_count: int, arity: int) -> int | None:
    """Number of fold levels iff `leaf_count == arity ** height` with at least
    one level — a tree no level pads. A regular tree folds with one scan; an
    irregular one (or a single leaf, which needs no fold) keeps the unrolled
    per-level-padding fold (see `MerkleTree._build`)."""
    height, size = 0, 1
    while size < leaf_count:
        size *= arity
        height += 1
    return height if size == leaf_count and height >= 1 else None


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

    `column_major` (keyword-only, default False) selects the COMMIT-side leaf
    layout: False reads a leaf as a matrix row (`commit` takes
    `[num_leaves, leaf_width]`), True as a matrix column (`[leaf_width,
    num_leaves]`), so a producer whose data is already column-per-leaf commits
    it without transposing to leaf-major. It changes only the commit leaf
    gather — `open`/`verify`/`reconstruct` are always leaf-major (one leaf per
    row), so a column-major consumer hands those the leaf-major matrix (the
    `commit` input's transpose).
    """

    def __init__(
        self,
        leaf_hasher: Sponge,
        compressor: Compression,
        *,
        column_major: bool = False,
    ) -> None:
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self.arity = compressor.arity
        self.digest_elems = compressor.chunk
        # Commit-side leaf layout (contract in the class docstring): when True
        # the leaf hash vmaps over axis 1 and the `merkle_commit` marker carries
        # `column_major` so the expander gathers columns. Touches commit only.
        self._column_major = column_major
        # Wrap the whole commit in the agnostic composite only when both blocks
        # lower to a hash-dedicated marker the vendor can expand; otherwise the
        # marker would be unexpandable, so fall back to the plain vmap path.
        self._fused = (
            leaf_hasher.has_dedicated_fusion and compressor.has_dedicated_fusion
        )

    # Value equality/hash for static jit-zone keys (#214) — identity equality
    # re-traces per instance. Both blocks compare by value themselves; the leaf
    # layout is part of the identity (a column-major tree traces a different
    # body), so it joins the key.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MerkleTree):
            return NotImplemented
        return (self._leaf_hasher, self._compressor, self._column_major) == (
            other._leaf_hasher,
            other._compressor,
            other._column_major,
        )

    def __hash__(self) -> int:
        return hash((self._leaf_hasher, self._compressor, self._column_major))

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a matrix: leaf-major `(num_leaves, leaf_width)`, or
        `(leaf_width, num_leaves)` when `column_major` (a leaf is a column).

        Returns `(raw_root (digest_elems,), digest_layers)`, where digest_layers
        runs leaf digests -> ... -> root, each (nodes_at_level, digest_elems)
        in the zero-padded form (see the module docstring).

        A binary tree keeps its historical power-of-two-height contract — its
        pad-free layout is what the fold-PCS query machinery indexes; k-ary
        consumers commit any height via the per-level padding.
        """
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
        # Field guard, but only when the leaf hasher names its field: `Sponge`
        # exposes `dtype`, while a consumer's custom leaf hash (the duck-typed
        # seam this commit's fallback serves) need not — so gate on its presence
        # rather than requiring it.
        leaf_dtype = getattr(self._leaf_hasher, "dtype", None)
        if leaf_dtype is not None and matrix.dtype != leaf_dtype:
            raise TypeError(
                f"leaf dtype {matrix.dtype} must match the hasher field {leaf_dtype}"
            )
        # Leaves are columns when column-major, else rows.
        num_leaves = matrix.shape[1] if self._column_major else matrix.shape[0]
        if self.arity == 2 and not is_power_of_two(num_leaves):
            raise ValueError(f"leaf count ({num_leaves}) must be a power of two")
        # The validation above guards eagerly, so it stays outside the marked
        # region; only `matrix` is an explicit operand (constants auto-lift).
        if self._fused:
            attrs = {"column_major": 1} if self._column_major else {}
            return fused_region(self._build, matrix, name=MERKLE_COMMIT_MARKER, **attrs)
        return self._build(matrix)

    # The #36 batched-permute dedup is opt-in: a leaf hasher / compressor that
    # provides the batched twin (zorch's Sponge / Compression) shares one lowered
    # permute body across the ragged fold layers; a block that only implements the
    # single-element `hash` / `compress` (e.g. a consumer's custom leaf hash like a
    # linear hash) keeps the plain vmap path — correct, just without the dedup.
    # Falling back here rather than requiring the twin keeps the leaf-hasher /
    # compressor seam duck-typed, so adding the dedup never breaks a consumer block.
    def _hash_leaves(self, matrix: Array) -> Array:
        hash_batched = getattr(self._leaf_hasher, "hash_batched", None)
        if hash_batched is not None:
            return hash_batched(matrix.T if self._column_major else matrix)
        return jax.vmap(self._leaf_hasher.hash, in_axes=1 if self._column_major else 0)(
            matrix
        )

    def _compress_groups(self, groups: Array) -> Array:
        compress_batched = getattr(self._compressor, "compress_batched", None)
        if compress_batched is not None:
            return compress_batched(groups)
        return jax.vmap(self._compressor.compress)(groups)

    def _build(self, matrix: Array, **_attrs: object) -> tuple[Array, list[Array]]:
        """The traced commit body: vmap the leaf hash, fold arity-sized groups
        per layer down to the root. Wrapped by `commit` as one
        `zorch.merkle_commit` region.

        `**_attrs` absorbs the marker's `composite.attributes` (threaded back by
        `lax.composite` on both the composite and inline paths); the body ignores
        them — `self._column_major` already fixes the layout.

        A binary tree (always power-of-two, so no level ever pads) folds with one
        `lax.scan`, tracing the per-layer compress once: the commit body is then
        O(1) in tree height, not `height` unrolled copies. The unrolled fold
        dominated the trace-commit compile, the same lever #163/#182 pulled on
        the verifier reconstruct. k-ary trees keep the unrolled fold — partly
        because per-level padding a fixed-shape scan can't express, and partly
        because the k-ary compress inside a loop crashes the CPU emitter
        (zkx#606); binary is the common, padding-free case."""
        # One batched leaf hash over the whole level (not a per-leaf vmap): a
        # dedicated-fusion permutation then shares one lowered permute body across
        # the ragged fold layers (see `_hash_leaves` for the vmap fallback).
        layer = self._hash_leaves(matrix)
        # Scan only the binary path; the height (always regular for a power-of-
        # two binary tree) is computed only there, never for the unrolled fold.
        if (
            self.arity == 2
            and (height := _regular_fold_height(layer.shape[0], self.arity)) is not None
        ):
            return self._fold_scan(layer, height)
        return self._fold_unrolled(layer)

    def _fold_unrolled(self, layer: Array) -> tuple[Array, list[Array]]:
        """Fold the leaf layer to the root one level at a time, completing a
        short top level with zero digests (the irregular-arity path; see
        `_build`). The padded form is stored, so an opening adjacent to a
        boundary reads its zero siblings from the layer like any others."""
        digest_layers = [layer]
        while layer.shape[0] > 1:
            rem = layer.shape[0] % self.arity
            if rem:
                pad = jnp.zeros((self.arity - rem, self.digest_elems), layer.dtype)
                layer = jnp.concatenate([layer, pad])
                digest_layers[-1] = layer
            groups = layer.reshape(-1, self.arity, self.digest_elems)
            layer = self._compress_groups(groups)
            digest_layers.append(layer)
        return digest_layers[-1][0], digest_layers

    def _fold_scan(self, leaf_layer: Array, height: int) -> tuple[Array, list[Array]]:
        """Fold a regular tree (`leaf_count == arity ** height`) with one scan.

        The scan carries a full-width working buffer whose live prefix shrinks by
        `arity` each level; one `compress` body is traced for all `height`
        levels. Each level's live prefix is the next `digest_layers` entry, so
        the per-level outputs are sliced back to the exact ragged shapes the
        unrolled fold produced — byte-identical, but O(1) in height. The buffer
        compresses its padding too; that waste only runs on a backend without the
        merkle-commit emitter (the GPU vendor replaces the whole region), and the
        sliced-off tail never reaches a digest layer."""
        n, a, d = leaf_layer.shape[0], self.arity, self.digest_elems

        def fold_level(buf: Array, _: None) -> tuple[Array, Array]:
            compressed = self._compress_groups(buf.reshape(n // a, a, d))
            return jnp.zeros_like(buf).at[: n // a].set(compressed), compressed

        _, stacked = jax.lax.scan(fold_level, leaf_layer, xs=None, length=height)
        digest_layers = [leaf_layer]
        for level in range(height):
            digest_layers.append(stacked[level, : n // a ** (level + 1)])
        return digest_layers[-1][0], digest_layers

    def open(
        self, matrix: Array, digest_layers: list[Array], index: int | Array
    ) -> Opening:
        """Authentication path for leaf `index`: its row plus each level's sibling.

        `matrix` is always leaf-major (`[num_leaves, leaf_width]`, one leaf per
        row), even when `column_major` (a commit-side-only flag): a column-major
        consumer passes the leaf-major transpose of its commit input here.

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
