"""MerkleTree.commit — structure, digest coherence, and open->verify roundtrip.

Correctness here is structural and does not need a golden vector: an independent
reconstruction of the root from each leaf digest plus its sibling path (using the
same compressor) must equal the committed root. Plonky3 merkle-root golden
vectors are added in the golden-vector slice.
"""

import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_params
from zorch.hash.sponge import Sponge, SpongeParams


def _kb16_tree(out=8, chunk=8):
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=out))
    comp = Compression(perm, CompressionParams(arity=2, chunk=chunk))
    return sponge, comp, MerkleTree(sponge, comp)


def _reconstruct_root(leaf_digest, leaf_idx, digest_layers, compressor):
    node, idx = leaf_digest, leaf_idx
    for level in range(len(digest_layers) - 1):  # leaf layer up to just below root
        sibling = digest_layers[level][idx ^ 1]
        pair = (
            jnp.stack([node, sibling]) if idx % 2 == 0 else jnp.stack([sibling, node])
        )
        node = compressor.compress(pair)
        idx //= 2
    return node


def test_commit_layer_shapes():
    _, _, tree = _kb16_tree()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4
    raw_root, layers = tree.commit(matrix)
    assert [l.shape for l in layers] == [(4, 8), (2, 8), (1, 8)]
    assert raw_root.shape == (8,)


def test_leaf_layer_is_per_row_sponge_hash():
    sponge, _, tree = _kb16_tree()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    _, layers = tree.commit(matrix)
    for i in range(4):
        assert jnp.array_equal(layers[0][i], sponge.hash(matrix[i]))


def test_open_verify_roundtrip_reconstructs_root():
    _, comp, tree = _kb16_tree()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    raw_root, layers = tree.commit(matrix)
    for i in range(4):
        rebuilt = _reconstruct_root(layers[0][i], i, layers, comp)
        assert jnp.array_equal(rebuilt, raw_root)


def test_single_row_root_is_its_leaf_digest():
    sponge, _, tree = _kb16_tree()
    matrix = jnp.arange(8, dtype=F).reshape(1, 8)  # height 1
    raw_root, layers = tree.commit(matrix)
    assert len(layers) == 1
    assert jnp.array_equal(raw_root, sponge.hash(matrix[0]))


def test_commit_deterministic():
    _, _, tree = _kb16_tree()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    r1, _ = tree.commit(matrix)
    r2, _ = tree.commit(matrix)
    assert jnp.array_equal(r1, r2)


def test_mismatched_digest_size_raises():
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=4))  # out 4
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))  # chunk 8
    try:
        MerkleTree(sponge, comp)
        assert False, "expected ValueError for out != chunk"
    except ValueError:
        pass


def test_non_power_of_arity_height_raises():
    _, _, tree = _kb16_tree()
    matrix = jnp.arange(24, dtype=F).reshape(3, 8)  # height 3, not a power of 2
    try:
        tree.commit(matrix)
        assert False, "expected ValueError for non-power-of-arity height"
    except ValueError:
        pass


if __name__ == "__main__":
    test_commit_layer_shapes()
    test_leaf_layer_is_per_row_sponge_hash()
    test_open_verify_roundtrip_reconstructs_root()
    test_single_row_root_is_its_leaf_digest()
    test_commit_deterministic()
    test_mismatched_digest_size_raises()
    test_non_power_of_arity_height_raises()
    print("ok")
