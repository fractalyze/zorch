"""Query-strided binary Merkle commitment — scheme-agnostic, on Sponge + Compression.

A plain Merkle tree pairs *adjacent* leaves (0-1, 2-3, …), so authenticating a
leaf opens one row and one sibling path. Some PCS query phases instead open a
whole *coset* of ``rows_per_query`` rows at once, and those rows sit a fixed
**stride** ``s = height / rows_per_query`` apart in the matrix
(``{i, i+s, i+2s, …}``) — not adjacent. A strided tree builds its bottom
``log2(rows_per_query)`` levels to pair at that stride,

    level l:  compress( prev[2x·s + y],  prev[(2x+1)·s + y] )  ->  next[x·s + y]

so after ``log2(rows_per_query)`` levels the ``rows_per_query`` rows of any query
have collapsed into one digest and exactly ``s`` such digests remain (one per
residue ``y = i mod s``). From there up it is an ordinary binary tree over those
``s`` nodes. One query then opens its ``rows_per_query`` rows
(``matrix[i :: s]``) under a single path from that first stored layer to the
root; the strided levels below are not stored — a verifier recomputes them by
re-hashing the opened rows. ``rows_per_query = 1`` adds no strided level and is
exactly the plain binary tree (`zorch.commit.merkle.MerkleTree`, arity 2).

Like `MerkleTree`, this emits no whole-tree marker: the nested permutes carry
their own dedicated `zorch.poseidon2` markers (a dedicated merkle-chain fusion
benchmarked slower than the per-permute kernels).

This carries the prover-side commitment and opening accessors plus the
verifier-side `open` / `reconstruct_root` (device-indexed, `vmap`-able), the
strided analogs of `MerkleTree.open` / `reconstruct_root` the query phase of a
folding PCS (e.g. WHIR) needs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from zorch.commit.merkle import MerkleTree, Opening
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import is_power_of_two, log2_strict_usize


class StridedMerkleTree:
    """A query-strided binary Merkle commitment over a single matrix.

    ``leaf_hasher`` squeezes each row to a ``digest_elems``-element leaf;
    ``compressor`` (arity 2) folds two digests into one. ``rows_per_query`` is the
    coset size one query opens — the count of strided bottom levels is
    ``log2(rows_per_query)``.
    """

    def __init__(
        self,
        leaf_hasher: Sponge,
        compressor: Compression,
        rows_per_query: int,
    ) -> None:
        if leaf_hasher.out != compressor.chunk:
            raise ValueError(
                f"leaf digest size ({leaf_hasher.out}) must equal compressor "
                f"chunk ({compressor.chunk})"
            )
        if compressor.arity != 2:
            raise ValueError(
                f"strided fold pairs leaves, so arity must be 2, got {compressor.arity}"
            )
        if not is_power_of_two(rows_per_query):
            raise ValueError(
                f"rows_per_query ({rows_per_query}) must be a power of two"
            )
        self._leaf_hasher = leaf_hasher
        self._compressor = compressor
        self._rows_per_query = rows_per_query
        self.digest_elems = compressor.chunk
        # The query layer up is a plain binary tree; reuse MerkleTree's scanned
        # regular fold rather than re-implement it.
        self._top = MerkleTree(leaf_hasher, compressor)

    # Value equality/hash for static jit-zone keys (#214) — identity equality
    # re-traces per instance. Both blocks compare by value themselves.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StridedMerkleTree):
            return NotImplemented
        return (self._leaf_hasher, self._compressor, self._rows_per_query) == (
            other._leaf_hasher,
            other._compressor,
            other._rows_per_query,
        )

    def __hash__(self) -> int:
        return hash((self._leaf_hasher, self._compressor, self._rows_per_query))

    def query_stride(self, height: int) -> int:
        """Spacing of one query's rows in a ``height``-row matrix — also the node
        count of the first stored (query) layer."""
        return height // self._rows_per_query

    def commit(self, matrix: Array) -> tuple[Array, list[Array]]:
        """Commit a ``(height, width)`` matrix.

        Returns ``(raw_root (digest_elems,), digest_layers)`` where
        ``digest_layers`` runs the query layer (``query_stride`` nodes) -> … ->
        root; the strided levels below the query layer are not stored.
        """
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
        height = matrix.shape[0]
        if not is_power_of_two(height):
            raise ValueError(f"matrix height ({height}) must be a power of two")
        if self._rows_per_query > height:
            raise ValueError(
                f"rows_per_query ({self._rows_per_query}) > leaves ({height})"
            )
        # No whole-tree marker (cf. MerkleTree: the dedicated merkle-chain fusion
        # is slower than per-permute poseidon2 kernels). The nested permutes carry
        # their own dedicated markers.
        return self._build(matrix, rows_per_query=self._rows_per_query)

    def _build(
        self, matrix: Array, *, rows_per_query: int
    ) -> tuple[Array, list[Array]]:
        """The commit body: vmap the leaf hash, fold ``log2(rows_per_query)``
        query-strided levels, then plain adjacent pairs to the root."""
        d = self.digest_elems
        layer = jax.vmap(self._leaf_hasher.hash)(matrix)
        query_stride = layer.shape[0] // rows_per_query

        # Query-strided levels (not stored): reshape (m, …) as (m/(2s), 2, s, d)
        # so lanes (2x·s + y, (2x+1)·s + y) pair, bring the pair axis adjacent,
        # and compress with the same vmap idiom the plain fold uses — the result
        # preserves next[x·s + y] order.
        for _ in range(log2_strict_usize(rows_per_query)):
            pairs = layer.reshape(-1, 2, query_stride, d).transpose(0, 2, 1, 3)
            layer = jax.vmap(self._compressor.compress)(pairs.reshape(-1, 2, d))

        # Plain binary fold from the query layer to the root (stored). The query
        # layer is a power-of-two node count, so reuse MerkleTree's scanned
        # regular-binary fold — O(1) in height and byte-identical to the unrolled
        # form (#221). A single query-layer node is already the root.
        if layer.shape[0] == 1:
            return layer[0], [layer]
        return self._top._fold_scan(layer, log2_strict_usize(layer.shape[0]))

    def opened_rows(self, matrix: Array, index: int) -> Array:
        """The ``rows_per_query`` rows query ``index`` opens —
        ``matrix[index :: query_stride]``, shape ``(rows_per_query, width)``."""
        stride = self.query_stride(matrix.shape[0])
        if not 0 <= index < stride:
            raise ValueError(f"index {index} out of range [0, {stride})")
        return matrix[index::stride]

    def query_merkle_proof(self, digest_layers: list[Array], index: int) -> Array:
        """Sibling digests from the query layer to just below the root,
        ``(proof_depth, digest_elems)`` — the strided levels below the query layer
        are recomputed by the verifier from the opened rows, so the proof starts
        at ``digest_layers[0]``."""
        stride = digest_layers[0].shape[0]
        if not 0 <= index < stride:
            raise ValueError(f"index {index} out of range [0, {stride})")
        siblings = []
        for layer in digest_layers[:-1]:
            siblings.append(layer[index ^ 1])
            index >>= 1
        return jnp.stack(siblings)

    def open(
        self, matrix: Array, digest_layers: list[Array], index: int | Array
    ) -> Opening:
        """Device-side opening of query ``index``: the opened coset
        ``matrix[index :: query_stride]`` `(rows_per_query, width)` plus the stored
        sibling path (query layer up to just below the root). The verifier-side
        analog of `opened_rows` + `query_merkle_proof`, but `index` may be a
        traced (device-sampled) value, so `vmap` over `index` opens a batch of
        queries. The strided levels below the query layer are NOT in the path —
        `reconstruct_root` recomputes them from the coset.

        Index validity is a prover-side precondition — enforced eagerly for a
        concrete index, skipped under tracing where `verify` owns out-of-range
        rejection (mirrors `MerkleTree.open`)."""
        stride = self.query_stride(matrix.shape[0])
        if not isinstance(index, jax.core.Tracer) and not 0 <= index < stride:
            raise IndexError(f"query index {index} out of range [0, {stride})")
        rows = matrix[index + stride * jnp.arange(self._rows_per_query)]
        path = []
        idx = index
        for layer in digest_layers[:-1]:  # query layer up to below root
            path.append(layer[idx ^ 1])
            idx = idx // 2
        return Opening(row=rows, path=path)

    def reconstruct_root(self, index: int | Array, opening: Opening) -> Array:
        """Rebuild the raw root from a strided `opening`. Collapse the opened coset
        (`opening.row`, `(rows_per_query, width)`) to its query-layer node by a
        plain adjacent-pair fold of the row hashes — that is how the unstored
        strided levels recombine one residue class — then climb `opening.path` (the
        stored siblings) by the query-index parity.

        Returns the root Array, not a verdict (a separator-binding consumer
        rebinds it before comparing). Single-index; batch by `vmap`-ing over
        `(index, opening)`. Mirrors `MerkleTree.reconstruct_root`."""
        # Collapse the coset to its query-layer node with the same scanned
        # regular-binary fold `_build` uses for the stored tree — one compress
        # body traced once (not unrolled per level), even under an outer vmap
        # (#163). `rows_per_query == 1` adds no strided level: the lone hashed
        # row is already the node (and `_fold_scan` can't take a zero-height
        # tree, mirroring `_build`'s single-node guard).
        leaves = jax.vmap(self._leaf_hasher.hash)(opening.row)  # (rows_per_query, d)
        if self._rows_per_query == 1:
            query_node = leaves[0]
        else:
            query_node, _ = self._top._fold_scan(
                leaves, log2_strict_usize(self._rows_per_query)
            )
        if not opening.path:
            return query_node

        def fold(
            carry: tuple[Array, Array], sibling: Array
        ) -> tuple[tuple[Array, Array], None]:
            return self._top._fold_with_sibling(*carry, sibling), None

        (root, _), _ = jax.lax.scan(
            fold, (query_node, jnp.asarray(index)), jnp.stack(opening.path)
        )
        return root
