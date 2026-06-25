"""MerkleTree.commit — structure, digest coherence, and open->verify roundtrip.

Correctness here is structural and does not need a golden vector: an independent
reconstruction of the root from each leaf digest plus its sibling path (using the
same compressor) must equal the committed root. Plonky3 merkle-root golden
vectors are added in the golden-vector slice.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import Array
from zk_dtypes import goldilocks_mont
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree, Opening
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.params import default_external_matrix
from zorch.hash.poseidon2.poseidon2 import POSEIDON2_MARKER, Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_POSEIDON2_ATTRS,
    koalabear16_params,
    koalabear16_perm,
)
from zorch.hash.sponge import Sponge, SpongeParams


def _non_standard_perm() -> Poseidon2:
    """A Poseidon2 whose external matrix is NOT the standard M4-circulant, so its
    permute carries only the generic fusion marker (`has_dedicated_fusion` False)
    rather than the dedicated `zorch.poseidon2` one."""
    params = koalabear16_params()
    perturbed = default_external_matrix(16, F).at[0, 0].add(jnp.ones((), F))
    return Poseidon2(dataclasses.replace(params, external_matrix=perturbed))


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

    def test_reconstruct_root_returns_committed_root(self) -> None:
        # reconstruct_root yields the raw root (not a verdict), so a
        # separator-binding consumer can rebind before comparing.
        tree, matrix, root, layers = _committed_4x8()
        for i in range(matrix.shape[0]):
            op = tree.open(matrix, layers, i)
            self.assertTrue(bool(jnp.array_equal(tree.reconstruct_root(i, op), root)))

    def test_opening_is_registered_pytree(self) -> None:
        # Opening must flatten to its leaves (row + each sibling) so vmap/jit can
        # carry it across the open->reconstruct boundary; an unregistered dataclass
        # flattens to a single opaque leaf.
        tree, matrix, _, layers = _committed_4x8()
        op = tree.open(matrix, layers, 1)  # height 4 -> 2-sibling path
        leaves, treedef = jax.tree_util.tree_flatten(op)
        self.assertEqual(len(leaves), 1 + len(op.path))
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(bool(jnp.array_equal(rebuilt.row, op.row)))

    def test_vmap_open_matches_per_index_open(self) -> None:
        # open is a single-index primitive; batching is jax.vmap over the index
        # (matrix/layers shared), exactly as commit vmaps the per-row leaf hash.
        tree, matrix, _, layers = _committed_4x8()
        indices = jnp.arange(matrix.shape[0])
        batched = jax.vmap(tree.open, in_axes=(None, None, 0))(matrix, layers, indices)
        self.assertEqual(batched.row.shape, (4, 8))
        self.assertEqual([p.shape for p in batched.path], [(4, 8), (4, 8)])
        for q in range(matrix.shape[0]):
            single = tree.open(matrix, layers, q)
            self.assertTrue(bool(jnp.array_equal(batched.row[q], single.row)))
            for lvl, sib in enumerate(single.path):
                self.assertTrue(bool(jnp.array_equal(batched.path[lvl][q], sib)))

    def test_vmap_reconstruct_root_recovers_committed_root(self) -> None:
        tree, matrix, root, layers = _committed_4x8()
        indices = jnp.arange(matrix.shape[0])
        openings = jax.vmap(tree.open, in_axes=(None, None, 0))(matrix, layers, indices)
        roots = jax.vmap(tree.reconstruct_root)(indices, openings)
        self.assertEqual(roots.shape, (4, 8))
        self.assertTrue(
            bool(jnp.all(jax.vmap(lambda r: jnp.array_equal(r, root))(roots)))
        )

    def test_jit_batched_open_reconstruct_roundtrip(self) -> None:
        # The whole batched open->reconstruct path traces under one jit (static
        # unroll, single region) — the property the fusion contract relies on.
        tree, matrix, root, layers = _committed_4x8()
        indices = jnp.arange(matrix.shape[0])

        @jax.jit
        def run(matrix: Array, layers: list[Array], indices: Array) -> Array:
            ops = jax.vmap(tree.open, in_axes=(None, None, 0))(matrix, layers, indices)
            return jax.vmap(tree.reconstruct_root)(indices, ops)

        roots = run(matrix, layers, indices)
        self.assertTrue(
            bool(jnp.all(jax.vmap(lambda r: jnp.array_equal(r, root))(roots)))
        )

    def test_commit_rolls_layer_fold_into_scan(self) -> None:
        # The prover commit folds the tree layers with one scan, not `log2(N)`
        # unrolled compresses: the lowered form carries a single while loop and
        # its composite count is independent of tree height, so commit's
        # trace+lower is O(1) in depth, not O(depth). The unrolled fold
        # dominated the trace-commit compile; the verifier reconstruct was
        # already rolled this way (#163/#182).
        _, _, tree = koalabear16_merkle()

        def lowered(height: int) -> str:
            matrix = jnp.zeros((height, 8), F)
            return jax.jit(tree.commit).lower(matrix).as_text()

        shallow, deep = lowered(4), lowered(64)  # depth 2 vs depth 6
        self.assertIn("stablehlo.while", shallow)  # the fold is a scan, not unrolled
        self.assertEqual(  # a deeper tree adds no unrolled region
            shallow.count("stablehlo.composite"), deep.count("stablehlo.composite")
        )

    def test_reconstruct_root_rolls_path_into_scan(self) -> None:
        # The leaf->root fold is one scan, not `depth` unrolled compresses: the
        # lowered form carries a single while loop and its op count is
        # independent of path depth, so the verifier's per-layer reconstruct
        # chains lower in O(1), not O(depth) (#163 — the unrolled fold dominated
        # trace+lower).
        _, _, tree = koalabear16_merkle()

        def lowered(depth: int) -> str:
            opening = Opening(
                row=jnp.zeros((8,), F),
                path=[jnp.zeros((8,), F) for _ in range(depth)],
            )
            return jax.jit(tree.reconstruct_root).lower(0, opening).as_text()

        shallow, deep = lowered(2), lowered(8)
        self.assertIn("stablehlo.while", shallow)  # the fold is a scan, not unrolled
        self.assertEqual(  # a 4x-deeper path adds no unrolled region
            shallow.count("stablehlo.composite"), deep.count("stablehlo.composite")
        )

    def test_reconstruct_root_recovers_root_on_deeper_tree(self) -> None:
        # The scan carries the leaf-index parity across levels; the other
        # roundtrips are depth-2, so exercise depth-4 to cover that propagation.
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(128, dtype=F).reshape(16, 8)  # height 16 -> depth 4
        root, layers = tree.commit(matrix)
        indices = jnp.arange(matrix.shape[0])
        openings = jax.vmap(tree.open, in_axes=(None, None, 0))(matrix, layers, indices)
        roots = jax.vmap(tree.reconstruct_root)(indices, openings)
        self.assertTrue(
            bool(jnp.all(jax.vmap(lambda r: jnp.array_equal(r, root))(roots)))
        )

    def test_reconstruct_roots_batches_mixed_depths(self) -> None:
        # The batched primitive rebuilds roots for trees of DIFFERENT height in
        # one vmap+scan: each path is zero-padded to the batch's max depth and a
        # validity mask makes the extra levels no-ops, so a depth-2 and a depth-4
        # opening reconstruct the same roots `reconstruct_root` gives per call
        # (#163 — the verifier's per-layer reconstruct loop folded into one pass).
        _, _, tree = koalabear16_merkle()
        deep_m = jnp.arange(128, dtype=F).reshape(16, 8)  # depth 4
        shallow_m = jnp.arange(32, dtype=F).reshape(4, 8)  # depth 2
        deep_root, deep_layers = tree.commit(deep_m)
        shallow_root, shallow_layers = tree.commit(shallow_m)
        legs = [
            (deep_root, 5, tree.open(deep_m, deep_layers, 5)),
            (deep_root, 12, tree.open(deep_m, deep_layers, 12)),
            (shallow_root, 1, tree.open(shallow_m, shallow_layers, 1)),
            (shallow_root, 3, tree.open(shallow_m, shallow_layers, 3)),
        ]
        max_depth = max(len(op.path) for _, _, op in legs)
        rows = jnp.stack([op.row for _, _, op in legs])
        indices = jnp.array([i for _, i, _ in legs])
        paths = jnp.stack(
            [
                jnp.pad(jnp.stack(op.path), ((0, max_depth - len(op.path)), (0, 0)))
                for _, _, op in legs
            ]
        )
        valid = jnp.stack([jnp.arange(max_depth) < len(op.path) for _, _, op in legs])

        roots = tree.reconstruct_roots(rows, indices, paths, valid)
        self.assertTrue(bool(jnp.all(roots == jnp.stack([r for r, _, _ in legs]))))
        for k, (_, idx, op) in enumerate(legs):  # identical to the per-call path
            self.assertTrue(
                bool(jnp.array_equal(roots[k], tree.reconstruct_root(idx, op)))
            )

    def test_reconstruct_roots_ignores_padding_levels(self) -> None:
        # A masked (padding) sibling must not change the rebuilt root — the
        # invariant that lets a shallow opening ride safely in a deeper batch.
        _, _, tree = koalabear16_merkle()
        m = jnp.arange(32, dtype=F).reshape(4, 8)  # depth 2
        root, layers = tree.commit(m)
        op = tree.open(m, layers, 2)
        max_depth = 4
        path = jnp.pad(jnp.stack(op.path), ((0, max_depth - len(op.path)), (0, 0)))
        valid = jnp.arange(max_depth) < len(op.path)
        tampered = path.at[len(op.path)].add(jnp.ones((), F))  # corrupt a pad level
        for p in (path, tampered):
            roots = tree.reconstruct_roots(
                op.row[None], jnp.array([2]), p[None], valid[None]
            )
            self.assertTrue(bool(jnp.array_equal(roots[0], root)))

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

    def test_verify_rejects_out_of_range_index(self) -> None:
        # Traced open does not bounds-check, so verify is the sole gate for an
        # untrusted/out-of-range index: reject below 0 and at >= 2^depth.
        tree, matrix, root, layers = _committed_4x8()
        op = tree.open(matrix, layers, 0)
        self.assertFalse(bool(tree.verify(root, -1, op)))
        self.assertFalse(bool(tree.verify(root, 1 << len(op.path), op)))

    def test_open_verify_single_leaf_empty_path(self) -> None:
        # height-1 tree: open's path is empty and verify reduces to leaf == root
        sponge, _, tree = koalabear16_merkle()
        matrix = jnp.arange(8, dtype=F).reshape(1, 8)
        root, layers = tree.commit(matrix)
        op = tree.open(matrix, layers, 0)
        self.assertEqual(op.path, [])
        self.assertTrue(bool(tree.verify(root, 0, op)))

    def test_open_rejects_out_of_range_index(self) -> None:
        # The eager guard fires for any concrete index — Python int or 0-d Array —
        # so a bad index raises instead of silently clamping the gather. Only a
        # tracer (vmap/jit) skips it; there verify owns the verdict.
        tree, matrix, _, layers = _committed_4x8()  # 4 leaves: valid 0..3
        for bad in (4, -1, jnp.asarray(4), jnp.asarray(-1)):
            with self.assertRaises(IndexError):
                tree.open(matrix, layers, bad)

    def test_blocks_expose_dedicated_fusion_capability(self) -> None:
        # The whole-commit composite gate: a block fuses to a dedicated kernel iff
        # its permutation lowers permute to a hash-named marker. Standard koalabear
        # poseidon2 qualifies; a non-standard external matrix does not.
        sponge, comp, _ = koalabear16_merkle()
        self.assertTrue(sponge.has_dedicated_fusion)
        self.assertTrue(comp.has_dedicated_fusion)
        perm = _non_standard_perm()
        self.assertFalse(Sponge(perm, SpongeParams(rate=8, out=8)).has_dedicated_fusion)
        self.assertFalse(
            Compression(perm, CompressionParams(arity=2, chunk=8)).has_dedicated_fusion
        )

    def test_commit_lowers_to_nested_poseidon2_markers(self) -> None:
        # commit emits no whole-tree marker; the leaf/fold permutes survive as
        # dedicated zorch.poseidon2 markers (with their attributes) the vendor
        # lowers to per-permute kernels. The permutes are vmap'd and their round
        # constants auto-lift, so this also guards composite.attributes survive.
        _, _, tree = koalabear16_merkle()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        text = jax.jit(tree.commit).lower(matrix).as_text()
        self.assertNotIn('"zorch.merkle_commit"', text)
        self.assertIn(f'"{POSEIDON2_MARKER}"', text)
        self.assertIn(KOALABEAR16_POSEIDON2_ATTRS, text)

    def test_value_equality_across_fresh_instances(self) -> None:
        # A tree seats in static jit-zone keys (#214): same-config builds must
        # compare and hash equal regardless of instance identity, and a config
        # change must break equality.
        _, _, a = koalabear16_merkle()
        _, _, b = koalabear16_merkle()
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        _, _, c = koalabear16_merkle(rate=4, out=4, chunk=4)
        self.assertNotEqual(a, c)


class KaryMerkleTreeTest(absltest.TestCase):
    """k-ary commit/open/reconstruct — structural, like the binary suite.

    The k-ary layout's byte-match against a concrete scheme (pil2-stark's
    arity-2/3/4 trees) lives in its consumer (zisk-zorch), keeping this suite
    scheme-agnostic."""

    def test_commit_layer_shapes_arity4(self) -> None:
        _, _, tree = koalabear16_merkle(out=4, arity=4, chunk=4)
        matrix = jnp.arange(128, dtype=F).reshape(16, 8)
        raw_root, layers = tree.commit(matrix)
        self.assertEqual([l.shape for l in layers], [(16, 4), (4, 4), (1, 4)])
        self.assertEqual(raw_root.shape, (4,))

    def test_open_verify_roundtrip_arity4(self) -> None:
        _, _, tree = koalabear16_merkle(out=4, arity=4, chunk=4)
        matrix = jnp.arange(128, dtype=F).reshape(16, 8)
        root, layers = tree.commit(matrix)
        for i in range(16):
            op = tree.open(matrix, layers, i)
            self.assertEqual([p.shape for p in op.path], [(3, 4), (3, 4)])
            self.assertTrue(bool(tree.verify(root, i, op)))

    def test_commit_pads_incomplete_top_level_with_zero_digests(self) -> None:
        # 2^5 leaves under arity 4 leave a 2-node top level; the level is
        # completed with zero digests (the padded form is stored), so the root
        # exists and an opening adjacent to the boundary still verifies.
        _, _, tree = koalabear16_merkle(out=4, arity=4, chunk=4)
        matrix = jnp.arange(32 * 8, dtype=F).reshape(32, 8)
        root, layers = tree.commit(matrix)
        self.assertEqual([l.shape for l in layers], [(32, 4), (8, 4), (4, 4), (1, 4)])
        zero = jnp.zeros((4,), F)
        self.assertTrue(bool(jnp.array_equal(layers[2][2], zero)))
        self.assertTrue(bool(jnp.array_equal(layers[2][3], zero)))
        for i in (0, 7, 31):
            self.assertTrue(bool(tree.verify(root, i, tree.open(matrix, layers, i))))

    def test_open_verify_roundtrip_arity3_with_padding(self) -> None:
        # Every level of a 4-leaf arity-3 tree pads (4 -> pad 6 -> 2 -> pad 3
        # -> 1), so the roundtrip covers sibling reads from both pad slots.
        _, _, tree = koalabear16_merkle(out=4, arity=3, chunk=4)
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        root, layers = tree.commit(matrix)
        for i in range(4):
            self.assertTrue(bool(tree.verify(root, i, tree.open(matrix, layers, i))))

    @absltest.skip(
        "zkx CPU mis-routes the batched k-ary compress permute to the binary "
        "poseidon2_merkle_compress kernel (shape mismatch crash); single-index "
        "reconstruct_root passes. Tracked as fractalyze/zkx#606."
    )
    def test_vmap_open_reconstruct_arity4(self) -> None:
        # The k-ary fold must trace under vmap like the binary one: a batched
        # open -> reconstruct_root over every leaf recovers the committed root.
        _, _, tree = koalabear16_merkle(out=4, arity=4, chunk=4)
        matrix = jnp.arange(128, dtype=F).reshape(16, 8)
        root, layers = tree.commit(matrix)
        indices = jnp.arange(16)
        openings = jax.vmap(tree.open, in_axes=(None, None, 0))(matrix, layers, indices)
        roots = jax.vmap(tree.reconstruct_root)(indices, openings)
        self.assertTrue(
            bool(jnp.all(jax.vmap(lambda r: jnp.array_equal(r, root))(roots)))
        )

    def test_reconstruct_roots_rejects_kary(self) -> None:
        _, _, tree = koalabear16_merkle(out=4, arity=4, chunk=4)
        with self.assertRaises(NotImplementedError):
            tree.reconstruct_roots(
                jnp.zeros((1, 8), F),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1, 2, 4), F),
                jnp.ones((1, 2), bool),
            )


class ColumnMajorMerkleTreeTest(absltest.TestCase):
    """`column_major=True` coverage: the commit equals a leaf-major commit of the
    transpose, the opening side stays leaf-major, and the flag joins the value
    identity. The layout contract itself lives in `MerkleTree`'s docs."""

    def _trees(self) -> tuple[MerkleTree, MerkleTree]:
        """A row-major and a column-major tree over the same blocks."""
        sponge, comp, row_tree = koalabear16_merkle()
        return row_tree, MerkleTree(sponge, comp, column_major=True)

    def test_commit_matches_leaf_major_transpose(self) -> None:
        # The core contract: hashing the columns of `[leaf_width, num_leaves]`
        # gives the same root + layers as the leaf-major commit of its transpose
        # (column r of the input == row r of the transpose == the same leaf).
        row_tree, col_tree = self._trees()
        leaf_major = jnp.arange(32, dtype=F).reshape(4, 8)  # 4 leaves x width 8
        row_root, row_layers = row_tree.commit(leaf_major)
        col_root, col_layers = col_tree.commit(leaf_major.T)  # [8, 4]
        self.assertTrue(bool(jnp.array_equal(col_root, row_root)))
        self.assertEqual([l.shape for l in col_layers], [l.shape for l in row_layers])
        for cl, rl in zip(col_layers, row_layers):
            self.assertTrue(bool(jnp.array_equal(cl, rl)))

    def test_open_verify_roundtrip_on_leaf_major(self) -> None:
        # open/verify are leaf-major regardless of the flag: the consumer commits
        # column-major but opens the leaf-major transpose, and the openings still
        # reconstruct the column-major root.
        _, col_tree = self._trees()
        leaf_major = jnp.arange(32, dtype=F).reshape(4, 8)
        col_root, col_layers = col_tree.commit(leaf_major.T)
        for i in range(4):
            op = col_tree.open(leaf_major, col_layers, i)
            self.assertTrue(bool(jnp.array_equal(op.row, leaf_major[i])))
            self.assertTrue(bool(col_tree.verify(col_root, i, op)))

    def test_non_power_of_two_num_leaves_raises(self) -> None:
        # The power-of-two guard counts leaves on axis 1 when column-major: a
        # `[leaf_width=8, num_leaves=3]` matrix has 3 leaves and must reject.
        _, col_tree = self._trees()
        with self.assertRaises(ValueError):
            col_tree.commit(jnp.arange(24, dtype=F).reshape(8, 3))

    def test_layout_is_part_of_value_identity(self) -> None:
        # The leaf layout is part of the jit-zone key (#214): a column-major tree
        # traces a different body, so it must not compare equal to the row-major
        # one, while two column-major builds stay equal.
        sponge, comp, row_tree = koalabear16_merkle()
        col_tree = MerkleTree(sponge, comp, column_major=True)
        col_tree2 = MerkleTree(sponge, comp, column_major=True)
        self.assertNotEqual(row_tree, col_tree)
        self.assertEqual(col_tree, col_tree2)
        self.assertEqual(hash(col_tree), hash(col_tree2))


class CommitDtypeRejectionTest(absltest.TestCase):
    """Committing a matrix whose field mismatches the leaf hasher is rejected —
    the permutation's dtype check fires during the leaf hash."""

    def test_commit_rejects_wrong_field_matrix(self) -> None:
        sponge, comp, _ = koalabear16_merkle()
        wrong_field = jnp.arange(8 * 8, dtype=goldilocks_mont).reshape(8, 8)
        with self.assertRaises(TypeError):
            MerkleTree(sponge, comp).commit(wrong_field)


if __name__ == "__main__":
    absltest.main()
