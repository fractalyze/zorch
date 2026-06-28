# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 single-matrix commitment (SMCS) over zorch's Merkle blocks.

Semantically SP1's ``CudaTcsProver::commit_tensors`` for one matrix of
power-of-two height: hash each row to a leaf, fold sibling pairs to a Merkle
root (zorch's ``MerkleTree`` over ``Sponge`` + ``Compression``), then apply
SP1's domain separator (the ``single_layer.rs`` convention) binding the matrix
shape into the root:

    commit = compress([merkle_root, sponge([log_height, width])])

The domain separator is SP1-specific and lives here, not in zorch — zorch's
``MerkleTree`` is deliberately scheme-agnostic and adds no separator. The same
holds for the verifier error codes, the open/verify path, and the heap proof
layout (``prove_openings_at_indices``): all SP1 glue, all here.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.commit.merkle import MerkleTree, Opening
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import log2_strict_usize


class VerifyCode(IntEnum):
    """SMCS verifier return codes.

    Values mirror SP1's verify enum so the FFI byte-match returns identical codes;
    1 (``WRONG_BATCH_SIZE``) is unreachable for a single-matrix scheme (no batch
    dimension), so it is omitted. ``verify_batch`` returns one of these as a traced
    ``int32`` rather than raising, so verification runs inside a jit/fused region.
    """

    OK = 0
    WRONG_HEIGHT = 2
    INDEX_OUT_OF_BOUNDS = 3
    ROOT_MISMATCH = 4


class SingleMatrixCommitmentScheme:
    """SP1's single-matrix commitment, built on zorch's agnostic Merkle blocks.

    Holds the leaf ``Sponge`` and the 2-to-1 ``Compression`` (both also drive the
    internal ``MerkleTree``); ``digest_elems`` is the compressor chunk size.
    """

    def __init__(self, sponge: Sponge, compressor: Compression):
        # Keep sponge + compressor too: the domain separator calls them directly
        # and zorch's MerkleTree does not expose its internals.
        #
        # column-major is a per-``commit`` choice, not a scheme property: the same
        # SMCS commits the trace codeword column-major (a leaf is a column) and the
        # FRI fold's pair-leaves row-major. Hold both trees; open/verify re-hash
        # extracted leaf rows, so they always use the row-major one.
        self._tree = MerkleTree(sponge, compressor)
        self._tree_column_major = MerkleTree(sponge, compressor, column_major=True)
        self._sponge = sponge
        self._compressor = compressor
        self.digest_elems = compressor.chunk

    def commit(
        self, matrix: Array, *, column_major: bool = False
    ) -> tuple[Array, list[Array]]:
        """Commit a base-field ``(height, width)`` matrix (power-of-two height).

        ``column_major`` commits the transpose ``(width, height)`` by hashing its
        columns (a leaf is a column), so the producer can hand a codeword in its
        native encode layout without a transpose; the root is identical to a
        row-major commit of the transpose. It is a per-call choice — the same
        scheme also commits row-major leaves (the FRI fold) — so it is not a
        constructor property; ``open``/``verify`` re-hash individually-extracted
        leaf rows and are layout-independent.

        Returns ``(commitment, digest_layers)``: the ``(digest_elems,)``
        commitment with SP1's domain separator applied, plus zorch's layered
        digest tree (leaf digests -> ... -> root). ``digest_layers`` is the
        prover data ``open_batch`` needs for sibling paths — zorch's
        ``MerkleTree`` is stateless, so the caller threads it back in (along
        with the matrix, which holds the openable rows) rather than holding a
        tree object.

        Extension-field matrices are not yet supported (SP1 commits each EF row as
        ``width * degree`` base-field elements, a reinterpretation not wired through
        zorch's blocks yet — the FFI byte-match slice, fractalyze/zorch#37). No
        explicit guard: the permutation's field check raises ``TypeError`` on an EF
        matrix ("state dtype ... must match the permutation field ...").
        """
        tree = self._tree_column_major if column_major else self._tree
        raw_root, digest_layers = tree.commit(matrix)
        # Column-major commit takes [width, height]; row-major takes [height, width].
        height, width = matrix.shape[::-1] if column_major else matrix.shape
        log_height = log2_strict_usize(height)  # power-of-two enforced by commit
        return self.bind_root(raw_root, log_height, width, matrix.dtype), digest_layers

    def bind_root(
        self, raw_root: Array, log_height: int, width: int, dtype: Any
    ) -> Array:
        """Apply SP1's domain separator to a raw root: the single source of the
        ``compress([root, sponge([log_height, width])])`` formula, shared by
        ``commit``, ``verify_batch``, and the stacked-open dual's raw-root wire
        pin so they can never drift. ``width`` is the base-field width
        (commit/verify both guard EF)."""
        params = self._sponge.hash(jnp.array([log_height, width], dtype=dtype))
        return self._compressor.compress(jnp.stack([raw_root, params]))

    def bind_structure(
        self, commitment: Array, row_counts: Array, column_counts: Array
    ) -> Array:
        """Bind jagged row/column structure into an SMCS commitment.

        SP1's jagged commit convention: hash ``[num_tables, row_counts...,
        column_counts...]`` and compress with the (already shape-bound)
        commitment, so the verifier's claimed chip layout is pinned by the
        commitment itself.
        """
        # A length mismatch wouldn't error downstream — it would hash a
        # malformed structure preimage silently, so fail loudly here.
        if row_counts.shape != column_counts.shape:
            raise ValueError(
                f"row_counts shape {row_counts.shape} must match "
                f"column_counts shape {column_counts.shape}"
            )
        num_tables = jnp.array([row_counts.shape[0]], dtype=row_counts.dtype)
        structure = jnp.concatenate([num_tables, row_counts, column_counts])
        return self._compressor.compress(
            jnp.stack([commitment, self._sponge.hash(structure)])
        )

    def open_batch(
        self, indices: Array, matrix: Array, digest_layers: list[Array]
    ) -> tuple[Array, list[Array]]:
        """Open the rows at ``indices`` and collect their Merkle sibling paths.

        Args:
            indices: 1-D ``(Q,)`` row indices; a single query passes a length-1
                array.
            matrix: the committed ``(height, width)`` matrix (holds the rows).
            digest_layers: ``commit``'s layered digest tree (leaf digests ->
                ... -> root).

        Returns ``(rows, proofs)``: ``rows`` is ``(Q, width)``; ``proofs`` is a
        list of length ``log_height`` whose i-th entry is ``(Q, digest_elems)``,
        the level-``i`` sibling digest of every query. The sibling gather is
        zorch's single-index ``MerkleTree.open`` batched over the queries with
        ``jax.vmap`` — the consumer keeps no Merkle path logic of its own.
        """
        opening = jax.vmap(self._tree.open, in_axes=(None, None, 0))(
            matrix, digest_layers, indices
        )
        return opening.row, opening.path

    def verify_batch(
        self,
        commitment: Array,
        dims: tuple[int, int],
        index: int,
        row: Array,
        proof: list[Array],
    ) -> Array:
        """Verify one opened row against an SMCS commitment.

        Reconstructs the raw root from ``row`` + sibling ``proof`` (zorch's
        ``reconstruct_root``), re-binds SP1's domain separator (via the same
        ``bind_root`` the prover used), and compares against ``commitment``.

        Args:
            commitment: the ``(digest_elems,)`` SMCS commitment.
            dims: ``(height, width)`` of the committed matrix.
            index: the opened row index.
            row: the opened ``(width,)`` row.
            proof: ``log_height`` sibling digests, leaf level first.

        Returns an ``int32`` :class:`VerifyCode`: ``OK`` iff the rebound root
        equals ``commitment``; ``WRONG_HEIGHT`` if ``proof`` has the wrong
        length; ``INDEX_OUT_OF_BOUNDS`` if ``index >= height``; else
        ``ROOT_MISMATCH``.
        """
        height, width = dims
        log_height = log2_strict_usize(height)
        if len(proof) != log_height:
            return jnp.array(VerifyCode.WRONG_HEIGHT, dtype=jnp.int32)

        # Reconstruct the raw root via zorch's fold (row + sibling path); the
        # consumer keeps only the SP1 separator rebind, not the generic Merkle
        # fold.
        raw_root = self._tree.reconstruct_root(index, Opening(row=row, path=proof))
        bound = self.bind_root(raw_root, log_height, width, row.dtype)
        matches = jnp.array_equal(bound, commitment)
        # Priority order: bounds first, then the reconstructed-root check.
        return jnp.where(
            (index < 0) | (index >= height),
            VerifyCode.INDEX_OUT_OF_BOUNDS,
            jnp.where(matches, VerifyCode.OK, VerifyCode.ROOT_MISMATCH),
        ).astype(jnp.int32)

    @staticmethod
    def heap_digests(digest_layers: list[Array]) -> Array:
        """Flatten zorch's layered digest tree into SP1's heap buffer.

        zorch returns layers leaf-first (``[leaves, ..., root]``); SP1's path
        kernel indexes a single ``(2N-1, digest_elems)`` array in heap order
        (root at 0, children of ``i`` at ``2i+1``/``2i+2``, leaf ``m`` at
        ``2^H-1+m``). Concatenating the layers root-first lays the levels down in
        exactly that order — the adapter is just the reversal + concat.
        """
        return jnp.concatenate(list(reversed(digest_layers)), axis=0)

    def prove_openings_at_indices(
        self, flat_digests: Array, indices: Array, tree_height: int
    ) -> Array:
        """SP1's ``computePaths`` kernel: sibling authentication paths from a
        heap-indexed digest buffer.

        Sibling indices are pure arithmetic on the heap layout (sibling of node
        ``j`` is ``((j-1) ^ 1) + 1``; parent is ``(j-1) >> 1``), so no level
        depends on the data of another — the whole path is one gather.

        Args:
            flat_digests: ``(2N-1, digest_elems)`` heap buffer (``heap_digests``).
            indices: 1-D ``(Q,)`` leaf indices.
            tree_height: ``log2(N)`` — the number of levels (static).

        Returns ``(Q, tree_height, digest_elems)``: per query, the sibling
        digest at each level, leaf level first.
        """
        leaf_offset = (1 << tree_height) - 1
        idx = indices + leaf_offset  # leaf nodes in the heap
        sibling_indices = []
        for _ in range(tree_height):
            sibling_indices.append(((idx - 1) ^ 1) + 1)
            idx = (idx - 1) >> 1  # ascend to the parent
        all_sibling_idx = jnp.stack(sibling_indices, axis=1)  # (Q, tree_height)
        return flat_digests[all_sibling_idx]
