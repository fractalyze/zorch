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

Like `MerkleTree`, the whole tree is wrapped in one `zorch.merkle_commit`
composite when both hash blocks lower to a dedicated fusion marker — but here the
marker also carries ``rows_per_query`` as a composite attribute, so a vendor's
expander knows the stride of the bottom levels. The wrapping passes only
``matrix`` (the round constants auto-lift), so this stays scheme-agnostic.

This is the prover-side commitment plus its opening accessors; reconstructing a
root from opened rows is the verifier's job and lives with the consuming scheme.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from zorch.commit.merkle import MERKLE_COMMIT_MARKER, MerkleTree
from zorch.fusion import fused_region
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
        *,
        fuse: bool = True,
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
        # Wrap only when both blocks lower to a hash-dedicated marker the vendor
        # can expand (cf. MerkleTree). `fuse=False` forces the byte-identical
        # inline path for a consumer whose vendor cannot expand a strided
        # merkle_commit.
        self._fused = (
            fuse
            and leaf_hasher.has_dedicated_fusion
            and compressor.has_dedicated_fusion
        )

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
        # The validation guards eagerly, so it stays outside the marked region;
        # only `matrix` is an explicit operand. `rows_per_query` rides as a
        # composite attribute (and is passed to `_build` as a keyword, so the
        # inline and marked paths are identical).
        if self._fused:
            return fused_region(
                self._build,
                matrix,
                name=MERKLE_COMMIT_MARKER,
                rows_per_query=self._rows_per_query,
            )
        return self._build(matrix, rows_per_query=self._rows_per_query)

    def _build(
        self, matrix: Array, *, rows_per_query: int
    ) -> tuple[Array, list[Array]]:
        """The traced commit body: vmap the leaf hash, fold ``log2(rows_per_query)``
        query-strided levels, then plain adjacent pairs to the root. Wrapped by
        ``commit`` as one ``zorch.merkle_commit`` region carrying
        ``rows_per_query``."""
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
