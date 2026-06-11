"""StridedMerkleTree.commit — plain-tree equivalence, strided structure, and an
opened-rows -> root reconstruction roundtrip.

Correctness is structural, no golden vector: rebuilding the root from one query's
opened rows (re-hashed and folded through the strided levels) plus its sibling
path must equal the committed root, and ``rows_per_query = 1`` must reproduce the
plain `MerkleTree` exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import Array
from zk_dtypes import koalabear_mont as F

from zorch._composite import _HAS_COMPOSITE_OP
from zorch.commit.merkle import MERKLE_COMMIT_MARKER, MerkleTree
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams


def _stack(rows_per_query: int) -> tuple[Sponge, Compression, StridedMerkleTree]:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return sponge, comp, StridedMerkleTree(sponge, comp, rows_per_query)


def _matrix(height: int, width: int = 3) -> Array:
    return (jnp.arange(height * width, dtype=F)).reshape(height, width)


def _reconstruct(
    sponge: Sponge, comp: Compression, opened: Array, path: Array, index: int
) -> Array:
    """Rebuild the root from one query's opened rows + sibling path: hash the
    ``rows_per_query`` rows, fold them (adjacent in coset order, which is how the
    strided levels collapse a single residue class) to the query-layer node, then
    fold that node up the plain path by leaf-index parity."""
    layer = jnp.stack([sponge.hash(opened[t]) for t in range(opened.shape[0])])
    while layer.shape[0] > 1:
        layer = jnp.stack(
            [
                comp.compress(layer[2 * x : 2 * x + 2])
                for x in range(layer.shape[0] // 2)
            ]
        )
    node, idx = layer[0], index
    for level in range(path.shape[0]):
        sib = path[level]
        pair = jnp.stack([node, sib]) if idx % 2 == 0 else jnp.stack([sib, node])
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
        self.assertTrue(bool(jnp.array_equal(s_root, p_root)))
        self.assertEqual(len(s_layers), len(p_layers))
        for sl, pl in zip(s_layers, p_layers):
            self.assertTrue(bool(jnp.array_equal(sl, pl)))

    def test_commit_layer_shapes(self) -> None:
        """height 16, rows_per_query 4 -> query_stride 4: stored layers start at
        the 4-node query layer (the two strided levels below are not stored)."""
        _, _, strided = _stack(rows_per_query=4)
        root, layers = strided.commit(_matrix(16))
        self.assertEqual([l.shape for l in layers], [(4, 8), (2, 8), (1, 8)])
        self.assertEqual(root.shape, (8,))
        self.assertEqual(strided.query_stride(16), 4)

    def test_opened_rows_are_the_strided_coset(self) -> None:
        _, _, strided = _stack(rows_per_query=4)
        matrix = _matrix(16)
        for index in range(4):  # query_stride
            opened = strided.opened_rows(matrix, index)
            self.assertEqual(opened.shape, (4, 3))
            self.assertTrue(bool(jnp.array_equal(opened, matrix[index::4])))

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
            self.assertTrue(bool(jnp.array_equal(rebuilt, root)), msg=f"query {index}")

    def test_single_query_layer_when_rows_per_query_equals_height(self) -> None:
        """rows_per_query == height collapses to a single query-layer node (the
        root); the strided levels fold the whole column, no path remains."""
        _, _, strided = _stack(rows_per_query=8)
        root, layers = strided.commit(_matrix(8))
        self.assertEqual([l.shape for l in layers], [(1, 8)])
        self.assertTrue(bool(jnp.array_equal(layers[-1][0], root)))

    @absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
    def test_commit_wraps_in_merkle_commit_marker(self) -> None:
        _, _, strided = _stack(rows_per_query=4)
        lowered = jax.jit(strided.commit).lower(_matrix(16)).as_text()
        self.assertIn("stablehlo.composite", lowered)
        self.assertIn(MERKLE_COMMIT_MARKER, lowered)


if __name__ == "__main__":
    absltest.main()
