"""StridedMerkleTree.commit — plain-tree equivalence, strided structure, and an
opened-rows -> root reconstruction roundtrip.

Correctness is structural, no golden vector: rebuilding the root from one query's
opened rows (re-hashed and folded through the strided levels) plus its sibling
path must equal the committed root, and ``rows_per_query = 1`` must reproduce the
plain `MerkleTree` exactly.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from absl.testing import absltest
from frx import Array
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree, Opening
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import POSEIDON2_MARKER
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import SPONGE_HASH_MARKER, Sponge, SpongeParams


def _stack(rows_per_query: int) -> tuple[Sponge, Compression, StridedMerkleTree]:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return sponge, comp, StridedMerkleTree(sponge, comp, rows_per_query)


def _matrix(height: int, width: int = 3) -> Array:
    return (fnp.arange(height * width, dtype=F)).reshape(height, width)


def _reconstruct(
    sponge: Sponge, comp: Compression, opened: Array, path: Array, index: int
) -> Array:
    """Rebuild the root from one query's opened rows + sibling path: hash the
    ``rows_per_query`` rows, fold them (adjacent in coset order, which is how the
    strided levels collapse a single residue class) to the query-layer node, then
    fold that node up the plain path by leaf-index parity."""
    layer = fnp.stack([sponge.hash(opened[t]) for t in range(opened.shape[0])])
    while layer.shape[0] > 1:
        layer = fnp.stack(
            [
                comp.compress(layer[2 * x : 2 * x + 2])
                for x in range(layer.shape[0] // 2)
            ]
        )
    node, idx = layer[0], index
    for level in range(path.shape[0]):
        sib = path[level]
        pair = fnp.stack([node, sib]) if idx % 2 == 0 else fnp.stack([sib, node])
        node, idx = comp.compress(pair), idx // 2
    return node


class StridedMerkleTest(absltest.TestCase):
    def test_rows_per_query_1_matches_plain_merkle(self) -> None:
        """No strided level — must be byte-identical to the plain binary tree."""
        sponge, comp, strided = _stack(rows_per_query=1)
        plain = MerkleTree(sponge, comp)
        matrix = _matrix(8)
        s_root, s_layers = strided.commit(matrix)
        p_root, p_layers = plain.commit(matrix)
        self.assertTrue(bool(fnp.array_equal(s_root, p_root)))
        self.assertEqual(len(s_layers), len(p_layers))
        for sl, pl in zip(s_layers, p_layers):
            self.assertTrue(bool(fnp.array_equal(sl, pl)))

    def test_commit_layer_shapes(self) -> None:
        """height 16, rows_per_query 4 -> query_stride 4: stored layers start at
        the 4-node query layer (the two strided levels below are not stored)."""
        _, _, strided = _stack(rows_per_query=4)
        root, layers = strided.commit(_matrix(16))
        self.assertEqual([l.shape for l in layers], [(4, 8), (2, 8), (1, 8)])
        self.assertEqual(root.shape, (8,))
        self.assertEqual(strided.query_stride(16), 4)

    def test_commit_lowers_to_nested_poseidon2_markers(self) -> None:
        # The query-strided `_build` is a separate path from MerkleTree's — cover
        # the same marker shape here too.
        _, _, strided = _stack(rows_per_query=4)
        text = frx.jit(strided.commit).lower(_matrix(16)).as_text()
        self.assertIn(f'"{SPONGE_HASH_MARKER}"', text)
        self.assertIn(f'"{POSEIDON2_MARKER}"', text)

    def test_opened_rows_are_the_strided_coset(self) -> None:
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        for index in range(4):  # query_stride
            opened = strided.opened_rows(matrix, index)
            self.assertEqual(opened.shape, (4, 3))
            self.assertTrue(bool(fnp.array_equal(opened, matrix[index::4])))

    def test_open_reconstruct_roundtrip(self) -> None:
        """Every query's opened rows + path must rebuild the committed root."""
        sponge, comp, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        root, layers = strided.commit(matrix)
        for index in range(strided.query_stride(16)):
            opened = strided.opened_rows(matrix, index)
            path = strided.query_merkle_proof(layers, index)
            self.assertEqual(path.shape, (2, 8))  # query layer (4) -> root: 2 levels
            rebuilt = _reconstruct(sponge, comp, opened, path, index)
            self.assertTrue(bool(fnp.array_equal(rebuilt, root)), msg=f"query {index}")

    def test_single_query_layer_when_rows_per_query_equals_height(self) -> None:
        """rows_per_query == height collapses to a single query-layer node (the
        root); the strided levels fold the whole column, no path remains."""
        _, _, strided = _stack(rows_per_query=8)
        root, layers = strided.commit(_matrix(8))
        self.assertEqual([l.shape for l in layers], [(1, 8)])
        self.assertTrue(bool(fnp.array_equal(layers[-1][0], root)))

    def test_device_open_matches_host_accessors(self) -> None:
        """`open` (device-indexed) returns the same coset + sibling path as the
        host `opened_rows` / `query_merkle_proof`."""
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        _, layers = strided.commit(matrix)
        for index in range(strided.query_stride(16)):
            opening = strided.open(matrix, layers, index)
            self.assertTrue(
                bool(fnp.array_equal(opening.row, strided.opened_rows(matrix, index)))
            )
            host_path = strided.query_merkle_proof(layers, index)
            self.assertTrue(bool(fnp.array_equal(fnp.stack(opening.path), host_path)))

    def test_device_reconstruct_root_roundtrip(self) -> None:
        """`open` -> `reconstruct_root` rebuilds the committed root for every query,
        across rows_per_query in {1, 4}."""
        for rpq in (1, 4):
            sponge, comp, strided = _stack(rows_per_query=rpq)
            matrix = _matrix(16)
            root, layers = strided.commit(matrix)
            for index in range(strided.query_stride(16)):
                opening = strided.open(matrix, layers, index)
                rebuilt = strided.reconstruct_root(index, opening)
                self.assertTrue(
                    bool(fnp.array_equal(rebuilt, root)), msg=f"rpq={rpq} q={index}"
                )

    def test_device_reconstruct_root_vmaps_over_traced_indices(self) -> None:
        """A batch of device-sampled query indices opens + reconstructs under one
        `jit`+`vmap` (the jit-clean query phase WHIR needs)."""
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        root, layers = strided.commit(matrix)
        indices = fnp.arange(strided.query_stride(16), dtype=fnp.int32)

        @frx.jit
        def open_and_rebuild(idx: Array) -> Array:
            opening = frx.vmap(lambda i: strided.open(matrix, layers, i))(idx)
            return frx.vmap(strided.reconstruct_root)(idx, opening)

        roots = open_and_rebuild(indices)
        self.assertEqual(roots.shape, (4, 8))
        for q in range(4):
            self.assertTrue(bool(fnp.array_equal(roots[q], root)), msg=f"q={q}")

    def test_reconstruct_root_rejects_tampered_row(self) -> None:
        """A corrupted opened coset must not rebuild the committed root."""
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        root, layers = strided.commit(matrix)
        opening = strided.open(matrix, layers, 1)
        tampered = Opening(
            row=opening.row.at[0, 0].add(fnp.ones((), F)), path=opening.path
        )
        rebuilt = strided.reconstruct_root(1, tampered)
        self.assertFalse(bool(fnp.array_equal(rebuilt, root)))

    def test_open_rejects_out_of_range_index(self) -> None:
        """A concrete query index outside [0, query_stride) trips the eager
        prover-side precondition (skipped only under tracing)."""
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        _, layers = strided.commit(matrix)
        with self.assertRaises(IndexError):
            strided.open(matrix, layers, strided.query_stride(16))

    def test_reconstruct_root_empty_path_when_rows_per_query_equals_height(
        self,
    ) -> None:
        """rows_per_query == height collapses the whole column to the root: the
        opening carries no path and reconstruct returns the query-layer node."""
        _, _, strided = _stack(rows_per_query=8)
        matrix = _matrix(8)
        root, layers = strided.commit(matrix)
        opening = strided.open(matrix, layers, 0)
        self.assertEqual(opening.path, [])
        rebuilt = strided.reconstruct_root(0, opening)
        self.assertTrue(bool(fnp.array_equal(rebuilt, root)))


if __name__ == "__main__":
    absltest.main()
