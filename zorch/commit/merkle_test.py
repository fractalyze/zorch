"""MerkleTree.commit — structure, digest coherence, and open->verify roundtrip.

Correctness here is structural and does not need a golden vector: an independent
reconstruction of the root from each leaf digest plus its sibling path (using the
same compressor) must equal the committed root. Plonky3 merkle-root golden
vectors are added in the golden-vector slice.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree
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


def _reconstruct_root(
    leaf_digest: Array,
    leaf_idx: int,
    digest_layers: Sequence[Array],
    compressor: Compression,
) -> Array:
    node, idx = leaf_digest, leaf_idx
    for level in range(len(digest_layers) - 1):  # leaf layer up to just below root
        sibling = digest_layers[level][idx ^ 1]
        pair = (
            jnp.stack([node, sibling]) if idx % 2 == 0 else jnp.stack([sibling, node])
        )
        node = compressor.compress(pair)
        idx //= 2
    return node


def test_commit_layer_shapes() -> None:
    _, _, tree = koalabear16_merkle()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4
    raw_root, layers = tree.commit(matrix)
    assert [l.shape for l in layers] == [(4, 8), (2, 8), (1, 8)]
    assert raw_root.shape == (8,)


def test_leaf_layer_is_per_row_sponge_hash() -> None:
    sponge, _, tree = koalabear16_merkle()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    _, layers = tree.commit(matrix)
    for i in range(4):
        assert jnp.array_equal(layers[0][i], sponge.hash(matrix[i]))


def test_open_verify_roundtrip_reconstructs_root() -> None:
    _, comp, tree = koalabear16_merkle()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    raw_root, layers = tree.commit(matrix)
    for i in range(4):
        rebuilt = _reconstruct_root(layers[0][i], i, layers, comp)
        assert jnp.array_equal(rebuilt, raw_root)


def test_commit_root_matches_plonky3_golden() -> None:
    _, _, tree = koalabear16_merkle()
    raw_root, _ = tree.commit(jnp.arange(32, dtype=F).reshape(4, 8))
    assert jnp.array_equal(raw_root, _PLONKY3_MERKLE_ROOT_4X8)


def test_single_row_root_is_its_leaf_digest() -> None:
    sponge, _, tree = koalabear16_merkle()
    matrix = jnp.arange(8, dtype=F).reshape(1, 8)  # height 1
    raw_root, layers = tree.commit(matrix)
    assert len(layers) == 1
    assert jnp.array_equal(raw_root, sponge.hash(matrix[0]))


def test_commit_deterministic() -> None:
    _, _, tree = koalabear16_merkle()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    r1, _ = tree.commit(matrix)
    r2, _ = tree.commit(matrix)
    assert jnp.array_equal(r1, r2)


def test_mismatched_digest_size_raises() -> None:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=4))  # out 4
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))  # chunk 8
    try:
        MerkleTree(sponge, comp)
        assert False, "expected ValueError for out != chunk"
    except ValueError:
        pass


def test_non_power_of_two_height_raises() -> None:
    _, _, tree = koalabear16_merkle()
    matrix = jnp.arange(24, dtype=F).reshape(3, 8)  # height 3, not a power of 2
    try:
        tree.commit(matrix)
        assert False, "expected ValueError for non-power-of-two height"
    except ValueError:
        pass


def test_non_2d_matrix_raises() -> None:
    _, _, tree = koalabear16_merkle()
    try:
        tree.commit(jnp.arange(8, dtype=F))  # 1-D, not a matrix
        assert False, "expected ValueError for non-2-D matrix"
    except ValueError:
        pass


def test_non_binary_compressor_raises() -> None:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=4, chunk=4))  # not 2-to-1
    try:
        MerkleTree(sponge, comp)
        assert False, "expected ValueError for non-binary compressor"
    except ValueError:
        pass


if __name__ == "__main__":
    test_commit_layer_shapes()
    test_leaf_layer_is_per_row_sponge_hash()
    test_open_verify_roundtrip_reconstructs_root()
    test_commit_root_matches_plonky3_golden()
    test_single_row_root_is_its_leaf_digest()
    test_commit_deterministic()
    test_mismatched_digest_size_raises()
    test_non_power_of_two_height_raises()
    test_non_2d_matrix_raises()
    test_non_binary_compressor_raises()
    print("ok")
