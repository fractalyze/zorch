"""Layer-by-layer k-ary Merkle commitment — scheme-agnostic, on Sponge + Compression.

`commit` hashes each matrix row to a leaf digest (Sponge), then folds sibling
groups per layer (Compression, whose `arity` sets the tree's) down to a single
root, returning `(raw_root, digest_layers)` (leaf digests first, root last).
It adds NO domain separator — that, the proof layout, and the verify error codes
are scheme-specific and live in the consumer (its commitment scheme).

A level whose node count is not a multiple of the arity is completed with
zero digests — the only convention that keeps a k-ary tree well-defined on
power-of-two heights (e.g. 2^k leaves under arity 4 leave a 2-node top level),
and the one the k-ary schemes this seam serves use. The padded form is what
`digest_layers` stores, so an opening near the boundary can read its zero
siblings like any others. A binary tree on a power-of-two height never pads,
so the arity-2 layout is exactly the historical one.

Each layer is one `vmap` over its nodes: an internal layer batches one
`compress` = one permute; the leaf layer batches one `hash`, which lowers as one
`zorch.sponge_hash` region per leaf (the whole rate-block absorb fused
into a single register-resident kernel, not a per-block permute chain). Those
collapse to one GPU kernel per node-batch once the permutation is captured to a
kernel (the poseidon2 fusion path, #25). The tree folds the layers one
right-sized level at a time (`_fold_to_root`) — see `_build` for why this beats a
full-width `scan`.

`fused_levels > 1` folds each run of whole levels as ONE `zorch.merkle_fold`
region (leaf-to-root layers returned together) instead of one `vmap` per level,
so a vendor emitter can keep the intermediate layers resident and delete the
per-level global read passes and launches — the port of flock-challenge's
`parent_hash3`. Every level's layer is still stored at its ordinary flat-tree
slot, so the fold stays byte-identical to the per-level one and `open` is
unchanged; absent the emitter the marker decomposes to the per-level fold.

`commit` lowers each leaf hash to a `zorch.sponge_hash` marker and each
fold layer's `compress` to a `zorch.poseidon2` permute marker, which the vendor
lowers to kernels directly. Committing by this plain vmap/fold body keeps the fast
per-permute kernels and lowers under symbolic dims for recompile-free export.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array

from zorch.fusion import fused_region
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import is_power_of_two

# The whole-region marker for a fused run of adjacent fold levels — one
# `frx.lax.composite` the vendor expands into a single kernel that keeps the
# intermediate layers register/shared-resident, writing each level to its
# ordinary flat-tree slot (so opening is unchanged). It is name-routed like
# `zorch.sponge_hash`: the region contains the per-level `zorch.poseidon2`
# permute markers nested inside, exactly the shape the `SpongeHashFusion`
# emitter already reads for the absorb chain. Absent that emitter the marker
# DECOMPOSES to the per-level fold — byte-identical, no launch change — so the
# seam is safe to carry ahead of the emitter. Attrs (`arity`, `digest_elems`,
# `levels`) are the structural metadata the emitter parses; the per-level node
# counts it needs to size shared memory come off the operand shape.
MERKLE_FOLD_MARKER = "zorch.merkle_fold"
MERKLE_FOLD_MARKER_VERSION = 0


@partial(frx.tree_util.register_dataclass, data_fields=["row", "path"], meta_fields=[])
@dataclass(frozen=True)
class Opening:
    """A single leaf's authentication path: the committed matrix `row` plus the
    sibling digest at each level (leaf-first, excluding the root).

    A pytree (leaves: `row` and each path sibling) so `open` / `reconstruct_root`
    batch under `frx.vmap` and trace under `jit`."""

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
        fused_levels: int = 1,
    ) -> None:
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        if fused_levels < 1:
            raise ValueError(f"fused_levels ({fused_levels}) must be >= 1")
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self.arity = compressor.arity
        self.digest_elems = compressor.chunk
        # Commit-side leaf layout (contract in the class docstring): when True
        # the leaf hash vmaps over axis 1 (a leaf is a column). Touches commit only.
        self._column_major = column_major
        # How many adjacent fold levels to fuse into one `zorch.merkle_fold`
        # region per dispatch (1 = the historical per-level fold, no marker).
        # The fold only marks a group when the compressor lowers to a dedicated
        # per-permute kernel — the region's value is nesting those kernels so the
        # vendor emitter can keep the intermediates resident; a non-dedicated
        # permutation gains nothing from the wrapper, so it stays per-level. The
        # marked and unmarked folds are byte-identical (the module docstring's
        # decompose-on-fallback), so `fused_levels` never changes the tree.
        self._fused_levels = fused_levels
        self._fused_fold = fused_levels > 1 and compressor.has_dedicated_fusion

    # Value equality/hash for static jit-zone keys (#214) — identity equality
    # re-traces per instance. Both blocks compare by value themselves; the leaf
    # layout is part of the identity (a column-major tree traces a different
    # body), so it joins the key.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MerkleTree):
            return NotImplemented
        return (
            self._leaf_hasher,
            self._compressor,
            self._column_major,
            self._fused_levels,
        ) == (
            other._leaf_hasher,
            other._compressor,
            other._column_major,
            other._fused_levels,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._leaf_hasher,
                self._compressor,
                self._column_major,
                self._fused_levels,
            )
        )

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a matrix: leaf-major `(num_leaves, leaf_width)`, or
        `(leaf_width, num_leaves)` when `column_major` (a leaf is a column).

        Returns `(raw_root (digest_elems,), digest_layers)`, where digest_layers
        runs leaf digests -> ... -> root, each (nodes_at_level, digest_elems)
        in the zero-padded form (see the module docstring).

        A binary tree keeps its historical power-of-two-height contract — its
        pad-free layout is what the fold-PCS query machinery indexes; k-ary
        consumers commit any height via the per-level padding.

        `hash_leaves` + `fold_digests` are this commit as two halves, for a
        consumer that cuts a jit-zone boundary between them (only the leaf
        hash's shapes carry the leaf width; see `zorch.pcs.jagged.commit`).
        """
        return self.fold_digests(self.hash_leaves(matrix))

    def hash_leaves(self, matrix: Array) -> Array:
        """Hash each leaf of `matrix` (layout per `column_major`, see `commit`)
        to its digest — the `(num_leaves, digest_elems)` leaf layer of
        `digest_layers`. Validation lives here, where the bad shape enters."""
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
        # Leaves are columns when column-major, else rows.
        num_leaves = matrix.shape[1] if self._column_major else matrix.shape[0]
        # An empty layer folds to nothing: the arity>2 path skips the pow2 gate,
        # then _fold_to_root reads its [0] leaf and raises deep inside the fold.
        if num_leaves == 0:
            raise ValueError("matrix must contain at least one leaf")
        if self.arity == 2 and not is_power_of_two(num_leaves):
            raise ValueError(f"leaf count ({num_leaves}) must be a power of two")
        return self._hash_leaves(matrix)

    def fold_digests(self, leaf_digests: Array) -> tuple[Array, list[Array]]:
        """Fold a `(num_leaves, digest_elems)` leaf-digest layer (from
        `hash_leaves`) to `(raw_root, digest_layers)` — `commit`'s second half.

        Compresses only each level's live nodes, one right-sized level at a
        time — a single `scan` would carry a full-width buffer and recompress
        the zero padding every level (~height× the work, the dominant commit
        runtime). The cost is an O(depth) compile: the per-level compresses are
        distinct shapes, so they don't share a cubin (#163 traded the other
        way); amortized once under the polymorphic compile-many-shards path.
        Leaf-layout-independent: `column_major` affects only the leaf hash."""
        # A jit-zone seam (the zoned commit folds in a separate zone from the
        # leaf hash): fail loud on a drifted leaf-digest shape here, not with a
        # cryptic reshape error inside _fold_to_root.
        if leaf_digests.ndim != 2 or leaf_digests.shape[-1] != self.digest_elems:
            raise ValueError(
                f"leaf_digests must be (num_leaves, {self.digest_elems}), got "
                f"shape {leaf_digests.shape}"
            )
        return self._fold_to_root(leaf_digests)

    # Batch the single-element leaf hash / compress with `vmap`: each op's
    # dedicated marker lowers identically batched (one shared decomposition), so
    # `vmap(single)` IS the batched kernel — a hand-written batched twin would
    # buy nothing.
    @partial(frx.jit, static_argnums=0)
    def _hash_leaves(self, matrix: Array) -> Array:
        return frx.vmap(self._leaf_hasher.hash, in_axes=1 if self._column_major else 0)(
            matrix
        )

    @partial(frx.jit, static_argnums=0)
    def _compress_groups(self, groups: Array) -> Array:
        return frx.vmap(self._compressor.compress)(groups)

    # `num_levels` is static (it fixes the unrolled decomposition + each output
    # shape), so it joins `self` in static_argnums.
    @partial(frx.jit, static_argnums=(0, 2))
    def _fold_levels(self, layer: Array, num_levels: int) -> tuple[Array, ...]:
        """Fold `num_levels` adjacent whole (no-pad) levels of `layer` as ONE
        `zorch.merkle_fold` region, returning each level's layer leaf-to-root.

        The decomposition runs the successive `compress` folds — each keeping its
        own `zorch.poseidon2` marker — so with no vendor merkle emitter the region
        decomposes to exactly the per-level fold (byte-identical), and with one it
        collapses to a single kernel holding the intermediates resident. Returning
        every level's layer (not just the last) is what keeps each node in its
        ordinary flat-tree slot, so `open` is unchanged. The caller guarantees the
        first `num_levels` folds from `layer.shape[0]` are each divisible by
        `arity` (no padding inside a fused group)."""

        def decomposition(lay: Array, **_attrs: object) -> tuple[Array, ...]:
            outs: list[Array] = []
            for _ in range(num_levels):
                groups = lay.reshape(-1, self.arity, self.digest_elems)
                lay = frx.vmap(self._compressor.compress)(groups)
                outs.append(lay)
            return tuple(outs)

        return fused_region(
            decomposition,
            layer,
            name=MERKLE_FOLD_MARKER,
            version=MERKLE_FOLD_MARKER_VERSION,
            arity=self.arity,
            digest_elems=self.digest_elems,
            levels=num_levels,
        )

    def _fusible_levels(self, n: int) -> int:
        """How many adjacent levels from `n` nodes fuse into one region: the
        largest `L <= fused_levels` for which every one of the `L` folds is a
        whole multiple of `arity` (no padding inside the group). `L < 2` means
        fall back to the padding-aware single-level step."""
        if not self._fused_fold:
            return 1
        levels = 0
        m = n
        while levels < self._fused_levels and m % self.arity == 0 and m >= self.arity:
            m //= self.arity
            levels += 1
            if m == 1:  # reached the root inside the group; can't fold further
                break
        return levels

    def _fold_one_level(self, layer: Array, digest_layers: list[Array]) -> Array:
        """One padding-aware fold level (the historical step): complete a short
        level with zero digests, storing the padded form in place (so an opening
        adjacent to a boundary reads its zero siblings), then compress."""
        rem = layer.shape[0] % self.arity
        if rem:
            pad = fnp.zeros((self.arity - rem, self.digest_elems), layer.dtype)
            layer = fnp.concatenate([layer, pad])
            digest_layers[-1] = layer
        groups = layer.reshape(-1, self.arity, self.digest_elems)
        return self._compress_groups(groups)

    def _fold_to_root(self, layer: Array) -> tuple[Array, list[Array]]:
        """Fold the leaf layer to the root, completing a short top level with zero
        digests (a binary power-of-two tree never pads; a k-ary tree may). The
        padded form is stored, so an opening adjacent to a boundary reads its zero
        siblings from the layer like any others.

        With `fused_levels > 1` (and a dedicated-fusion compressor) each run of
        whole levels folds as one `zorch.merkle_fold` region; the padding tail and
        any group too short to fuse fall back to the per-level step. Every level's
        layer is stored either way, so the two paths are byte-identical and share
        one `open`/`reconstruct`."""
        digest_layers = [layer]
        while layer.shape[0] > 1:
            num_levels = self._fusible_levels(layer.shape[0])
            if num_levels >= 2:
                fused = self._fold_levels(layer, num_levels)
                digest_layers.extend(fused)
                layer = fused[-1]
            else:
                layer = self._fold_one_level(layer, digest_layers)
                digest_layers.append(layer)
        return digest_layers[-1][0], digest_layers

    def open(
        self, matrix: Array, digest_layers: list[Array], index: int | Array
    ) -> Opening:
        """Authentication path for leaf `index`: its row plus each level's sibling.

        `matrix` is always leaf-major (`[num_leaves, leaf_width]`, one leaf per
        row), even when `column_major` (a commit-side-only flag): a column-major
        consumer passes the leaf-major transpose of its commit input here.

        Single-index by construction; batch by `frx.vmap`-ing over `index` (the
        sibling gather is orchestration outside the fused permute, like `commit`).
        Index validity is a prover-side precondition — enforced eagerly for any
        concrete index (Python int or 0-d Array), skipped only under tracing,
        where the value is unknown and JAX would silently clamp an out-of-range
        gather; there `verify` owns out-of-range rejection.
        """
        if not isinstance(index, frx.core.Tracer) and not 0 <= index < matrix.shape[0]:
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
                j = fnp.arange(self.arity - 1)
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
        left = fnp.where(is_left, node, sibling)
        right = fnp.where(is_left, sibling, node)
        return self._compressor.compress(fnp.stack([left, right])), idx // 2

    def _fold_with_siblings(
        self, node: Array, idx: Array, siblings: Array
    ) -> tuple[Array, Array]:
        """k-ary dual of `_fold_with_sibling`: re-insert `node` at its group
        position among `siblings` `(arity-1, digest_elems)` by data-select (no
        Python branch, so the fold traces under `vmap`), compress the group,
        and return the parent with `idx` divided for the level above."""
        pos = idx % self.arity
        i = fnp.arange(self.arity)
        # Slot i holds the sibling that `open` packed for it (skipping the
        # node's own slot), except slot pos, which holds the node.
        gathered = siblings[fnp.clip(i - (i > pos), 0, self.arity - 2)]
        group = fnp.where((i == pos)[:, None], node[None, :], gathered)
        return self._compressor.compress(group), idx // self.arity

    def reconstruct_root(self, index: int | Array, opening: Opening) -> Array:
        """Rebuild the raw root from an `opening`'s row + path (leaf-first).

        Returns the root Array, not a verdict — a separator-binding consumer
        (e.g. SP1's SMCS) rebinds the raw root before comparing, which
        `verify`'s plain equality can't express. Single-index; batch by
        `frx.vmap`-ing over `(index, opening)`."""
        node = self._leaf_hasher.hash(opening.row)
        # Unroll the leaf->root fold over the static-depth path instead of a
        # `scan`: a scan lowers to a `while` whose custom poseidon2 fusion body
        # miscompiles the s32 index carry under @jit on the sponge plugin (wrong
        # parity -> wrong pair order -> wrong root; eager is correct), while each
        # level's compress is one same-shape pair, so the reuse_key dedups them to
        # one cubin and the trace stays small. Knowledge:
        # zorch-jagged-verify-jit-miscompiles-merkle-fold.
        idx = index
        for siblings in opening.path:
            if self.arity == 2:
                node, idx = self._fold_with_sibling(node, idx, siblings)
            else:
                node, idx = self._fold_with_siblings(node, idx, siblings)
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
                node = fnp.where(active, folded, node)
                return (node, idx), None

            (node, _), _ = frx.lax.scan(fold, (node, index), (path, mask))
            return node

        return frx.vmap(one)(rows, indices, paths, valid)

    def verify(self, root: Array, index: int, opening: Opening) -> bool:
        """Rebuild the root from the row + path; compare to the committed root."""
        if not 0 <= index < self.arity ** len(opening.path):
            return False
        return bool(fnp.array_equal(self.reconstruct_root(index, opening), root))
