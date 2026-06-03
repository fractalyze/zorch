"""MerkleTree.commit — structure, digest coherence, and open->verify roundtrip.

Correctness here is structural and does not need a golden vector: an independent
reconstruction of the root from each leaf digest plus its sibling path (using the
same compressor) must equal the committed root. Plonky3 merkle-root golden
vectors are added in the golden-vector slice.
"""

from __future__ import annotations

import jax.numpy as jnp
from absl.testing import absltest
from jax import Array
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree, Opening
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams

# Plonky3 golden vector (p3_commit=4318eba..., default_koalabear_poseidon2_16):
# PaddingFreeSponge<_,16,8,8> leaves + TruncatedPermutation<_,2,8,16> over
# arange(32) reshaped to a 4x8 matrix (hash rows, fold pairs).
_PLONKY3_MERKLE_ROOT_4X8 = jnp.array(
    [
        1670701318,
        437280557,
        23464423,
        637192971,
        1642004034,
        359231982,
        157670030,
        587973557,
    ],
    dtype=F,
)


def _committed_4x8() -> tuple[MerkleTree, Array, Array, list[Array]]:
    """Commit a 4x8 koalabear matrix; return (tree, matrix, root, layers)."""
    _, _, tree = koalabear16_merkle()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    root, layers = tree.commit(matrix)
    return tree, matrix, root, layers


class MerkleTreeTest(absltest.TestCase):
    def test_commit_layer_shapes(self) -> None:
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4
        raw_root, layers = tree.commit(matrix)
        self.assertEqual([l.shape for l in layers], [(4, 8), (2, 8), (1, 8)])
        self.assertEqual(raw_root.shape, (8,))

    def test_leaf_layer_is_per_row_sponge_hash(self) -> None:
        sponge, _, tree = koalabear16_merkle()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        _, layers = tree.commit(matrix)
        for i in range(4):
            self.assertTrue(bool(jnp.array_equal(layers[0][i], sponge.hash(matrix[i]))))

    def test_open_verify_roundtrip_reconstructs_root(self) -> None:
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        root, layers = tree.commit(matrix)
        for i in range(4):
            op = tree.open(matrix, layers, i)
            self.assertTrue(bool(jnp.array_equal(op.row, matrix[i])))
            self.assertTrue(bool(tree.verify(root, i, op)))

    def test_commit_root_matches_plonky3_golden(self) -> None:
        _, _, tree = koalabear16_merkle()
        raw_root, _ = tree.commit(jnp.arange(32, dtype=F).reshape(4, 8))
        self.assertTrue(bool(jnp.array_equal(raw_root, _PLONKY3_MERKLE_ROOT_4X8)))

    def test_single_row_root_is_its_leaf_digest(self) -> None:
        sponge, _, tree = koalabear16_merkle()
        matrix = jnp.arange(8, dtype=F).reshape(1, 8)  # height 1
        raw_root, layers = tree.commit(matrix)
        self.assertEqual(len(layers), 1)
        self.assertTrue(bool(jnp.array_equal(raw_root, sponge.hash(matrix[0]))))

    def test_commit_deterministic(self) -> None:
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        r1, _ = tree.commit(matrix)
        r2, _ = tree.commit(matrix)
        self.assertTrue(bool(jnp.array_equal(r1, r2)))

    def test_mismatched_digest_size_raises(self) -> None:
        perm = koalabear16_perm()
        sponge = Sponge(perm, SpongeParams(rate=8, out=4))  # out 4
        comp = Compression(perm, CompressionParams(arity=2, chunk=8))  # chunk 8
        with self.assertRaises(ValueError):
            MerkleTree(sponge, comp)

    def test_non_power_of_two_height_raises(self) -> None:
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(24, dtype=F).reshape(3, 8)  # height 3, not a power of 2
        with self.assertRaises(ValueError):
            tree.commit(matrix)

    def test_non_2d_matrix_raises(self) -> None:
        _, _, tree = koalabear16_merkle()
        with self.assertRaises(ValueError):
            tree.commit(jnp.arange(8, dtype=F))  # 1-D, not a matrix

    def test_non_binary_compressor_raises(self) -> None:
        perm = koalabear16_perm()
        sponge = Sponge(perm, SpongeParams(rate=8, out=8))
        comp = Compression(perm, CompressionParams(arity=4, chunk=4))  # not 2-to-1
        with self.assertRaises(ValueError):
            MerkleTree(sponge, comp)

    def test_verify_rejects_tampered_row(self) -> None:
        tree, matrix, root, layers = _committed_4x8()
        op = tree.open(matrix, layers, 2)
        bad = Opening(row=op.row.at[0].add(jnp.ones((), F)), path=op.path)
        self.assertFalse(bool(tree.verify(root, 2, bad)))

    def test_verify_rejects_tampered_path(self) -> None:
        tree, matrix, root, layers = _committed_4x8()
        op = tree.open(matrix, layers, 1)
        bad_path = list(op.path)
        bad_path[0] = bad_path[0].at[0].add(jnp.ones((), F))
        self.assertFalse(bool(tree.verify(root, 1, Opening(row=op.row, path=bad_path))))

    def test_verify_rejects_wrong_index(self) -> None:
        tree, matrix, root, layers = _committed_4x8()
        op = tree.open(matrix, layers, 0)  # opening for leaf 0
        self.assertFalse(bool(tree.verify(root, 1, op)))  # wrong index -> reject

    def test_open_verify_single_leaf_empty_path(self) -> None:
        # height-1 tree: open's path is empty and verify reduces to leaf == root
        sponge, _, tree = koalabear16_merkle()
        matrix = jnp.arange(8, dtype=F).reshape(1, 8)
        root, layers = tree.commit(matrix)
        op = tree.open(matrix, layers, 0)
        self.assertEqual(op.path, [])
        self.assertTrue(bool(tree.verify(root, 0, op)))

    def test_open_rejects_out_of_range_index(self) -> None:
        tree, matrix, _, layers = _committed_4x8()  # 4 leaves: valid 0..3
        for bad in (4, -1):
            with self.assertRaises(IndexError):
                tree.open(matrix, layers, bad)


if __name__ == "__main__":
    absltest.main()
