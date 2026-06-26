# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SMCS.commit — Merkle root + SP1 domain separator.

The raw Merkle root and the domain-separated commitment are pinned to SP1's
koalabear16 parameterization (the powers-of-two internal diagonal, distinct
from the Plonky3 instance zorch's merkle_test uses): an independent
recomputation of SP1's separator formula, plus pinned regression vectors. SP1
byte-match equivalence against the reference prover is the FFI slice.
"""

import jax.numpy as jnp
from absl.testing import absltest
from jax import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.commit.merkle import MerkleTree
from zorch.commit.smcs import SingleMatrixCommitmentScheme, VerifyCode
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

# SP1-form koalabear16 root over arange(32).reshape(4, 8) — before the
# separator.
_SP1_RAW_ROOT_4X8 = jnp.array(
    [
        709053809,
        548310247,
        1460186906,
        135994348,
        1863522735,
        953629012,
        708601688,
        648442714,
    ],
    dtype=F,
)
# commit() over the same matrix: compress([root, sponge([log_height=2, width=8])]).
_SMCS_COMMIT_4X8 = jnp.array(
    [
        1353110693,
        797241874,
        1199727970,
        1276263656,
        166408803,
        91315187,
        1416168646,
        20384573,
    ],
    dtype=F,
)


def _smcs() -> tuple[Sponge, Compression, SingleMatrixCommitmentScheme]:
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return sponge, comp, SingleMatrixCommitmentScheme(sponge, comp)


def _proof_for(proofs: list[Array], q: int) -> list[Array]:
    """Slice the per-level batched siblings down to query ``q``'s single path."""
    return [proofs[level][q] for level in range(len(proofs))]


class SingleMatrixCommitmentSchemeTest(absltest.TestCase):
    def test_commit_applies_sp1_domain_separator(self) -> None:
        sponge, comp, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        committed, _ = smcs.commit(matrix)

        raw_root, _ = MerkleTree(sponge, comp).commit(matrix)
        self.assertTrue(bool(jnp.array_equal(raw_root, _SP1_RAW_ROOT_4X8)))

        # Independent recomputation of SP1's separator over the golden root.
        params = sponge.hash(jnp.array([2, 8], dtype=F))  # [log_height, width]
        expected = comp.compress(jnp.stack([raw_root, params]))
        self.assertTrue(bool(jnp.array_equal(committed, expected)))
        self.assertTrue(bool(jnp.array_equal(committed, _SMCS_COMMIT_4X8)))

    def test_commit_returns_commitment_and_prover_layers(self) -> None:
        # commit hands back the layered digest tree (zorch's stateless prover
        # data) alongside the commitment, so open_batch can produce sibling paths
        # without recommitting. Layers run leaf-digests -> ... -> root.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4 -> 3 layers
        committed, digest_layers = smcs.commit(matrix)
        self.assertEqual(committed.shape, (8,))
        self.assertEqual(
            [layer.shape for layer in digest_layers], [(4, 8), (2, 8), (1, 8)]
        )
        self.assertTrue(bool(jnp.array_equal(committed, _SMCS_COMMIT_4X8)))

    def test_open_batch_returns_rows_and_sibling_path(self) -> None:
        # open_batch returns the queried rows plus, per Merkle layer, the sibling
        # digest of each query's node — log_height layers, batched over the queries.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4 -> log_height 2
        _, layers = smcs.commit(matrix)
        indices = jnp.array([0, 3])

        rows, proofs = smcs.open_batch(indices, matrix, layers)

        self.assertEqual(rows.shape, (2, 8))
        self.assertTrue(bool(jnp.array_equal(rows[0], matrix[0])))
        self.assertTrue(bool(jnp.array_equal(rows[1], matrix[3])))

        self.assertLen(proofs, 2)  # log_height
        # Level 0 sibling is the other leaf digest; level 1 sibling is the opposite
        # subtree root. Selected by flipping the low bit of the per-level index.
        self.assertTrue(bool(jnp.array_equal(proofs[0], layers[0][indices ^ 1])))
        self.assertTrue(bool(jnp.array_equal(proofs[1], layers[1][(indices >> 1) ^ 1])))

    def test_open_verify_roundtrip_ok_for_every_index(self) -> None:
        # Opening any row and folding its sibling path back up must reconstruct the
        # bound commitment exactly -> OK. Structural, no golden needed.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        commitment, layers = smcs.commit(matrix)
        indices = jnp.array([0, 1, 2, 3])
        rows, proofs = smcs.open_batch(indices, matrix, layers)
        for q, idx in enumerate(range(4)):
            code = smcs.verify_batch(
                commitment, (4, 8), idx, rows[q], _proof_for(proofs, q)
            )
            self.assertEqual(int(code), VerifyCode.OK)

    def test_verify_rejects_tampered_row(self) -> None:
        # A corrupted opened row re-hashes to a different leaf -> the rebound root
        # cannot match the commitment.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        commitment, layers = smcs.commit(matrix)
        rows, proofs = smcs.open_batch(jnp.array([1]), matrix, layers)
        tampered = rows[0] + jnp.ones(8, dtype=F)
        code = smcs.verify_batch(commitment, (4, 8), 1, tampered, _proof_for(proofs, 0))
        self.assertEqual(int(code), VerifyCode.ROOT_MISMATCH)

    def test_verify_rejects_index_out_of_bounds(self) -> None:
        # index >= height is caught even with a correctly-sized proof.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)
        commitment, layers = smcs.commit(matrix)
        rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
        code = smcs.verify_batch(commitment, (4, 8), 4, rows[0], _proof_for(proofs, 0))
        self.assertEqual(int(code), VerifyCode.INDEX_OUT_OF_BOUNDS)

    def test_verify_rejects_wrong_proof_length(self) -> None:
        # A proof whose length != log_height cannot authenticate any leaf.
        _, _, smcs = _smcs()
        matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # log_height 2
        commitment, layers = smcs.commit(matrix)
        rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
        short_proof = _proof_for(proofs, 0)[:1]  # 1 sibling, expected 2
        code = smcs.verify_batch(commitment, (4, 8), 0, rows[0], short_proof)
        self.assertEqual(int(code), VerifyCode.WRONG_HEIGHT)

    def test_heap_digests_layout(self) -> None:
        # The layered tree flattens to SP1's heap buffer: root at index 0, then each
        # level top-down, leaves last. For 2^H leaves the buffer has 2^(H+1)-1 nodes.
        _, _, smcs = _smcs()
        matrix = jnp.arange(64, dtype=F).reshape(8, 8)  # height 8 -> 15-node heap
        _, layers = smcs.commit(matrix)
        heap = smcs.heap_digests(layers)
        self.assertEqual(heap.shape, (15, 8))
        self.assertTrue(bool(jnp.array_equal(heap[0], layers[-1][0])))  # root at 0
        self.assertTrue(bool(jnp.array_equal(heap[7:], layers[0])))  # leaves last

    def test_prove_openings_matches_open_batch_siblings(self) -> None:
        # The heap-indexed path kernel must select exactly the siblings that the
        # layered open_batch reads — same authentication path, two representations.
        _, _, smcs = _smcs()
        matrix = jnp.arange(64, dtype=F).reshape(8, 8)  # height 8, tree_height 3
        _, layers = smcs.commit(matrix)
        heap = smcs.heap_digests(layers)
        indices = jnp.array([0, 3, 5])

        paths = smcs.prove_openings_at_indices(heap, indices, 3)
        self.assertEqual(paths.shape, (3, 3, 8))  # (Q, tree_height, digest_elems)

        _, proofs = smcs.open_batch(indices, matrix, layers)
        for level in range(3):
            self.assertTrue(bool(jnp.array_equal(paths[:, level, :], proofs[level])))

    def test_open_verify_single_row_roundtrip(self) -> None:
        # height 1 boundary: log_height 0 -> empty proof; verify folds nothing and
        # just rebinds the lone leaf digest.
        _, _, smcs = _smcs()
        matrix = jnp.arange(8, dtype=F).reshape(1, 8)
        commitment, layers = smcs.commit(matrix)
        rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
        self.assertEqual(proofs, [])
        code = smcs.verify_batch(commitment, (1, 8), 0, rows[0], _proof_for(proofs, 0))
        self.assertEqual(int(code), VerifyCode.OK)

    def test_commit_shape_and_determinism(self) -> None:
        _, _, smcs = _smcs()
        m = jnp.arange(32, dtype=F).reshape(4, 8)
        (c1, _), (c2, _) = smcs.commit(m), smcs.commit(m)
        self.assertEqual(c1.shape, (8,))
        self.assertTrue(bool(jnp.array_equal(c1, c2)))

    def test_single_row_commit_log_height_zero(self) -> None:
        _, _, smcs = _smcs()
        # height 1 -> log_height 0; raw root is the lone leaf digest, then bound.
        c, _ = smcs.commit(jnp.arange(8, dtype=F).reshape(1, 8))
        self.assertEqual(c.shape, (8,))

    def test_non_power_of_two_height_raises(self) -> None:
        _, _, smcs = _smcs()
        with self.assertRaises(ValueError):
            smcs.commit(jnp.arange(24, dtype=F).reshape(3, 8))  # height 3

    def test_extension_field_matrix_not_yet_supported(self) -> None:
        # EF commit (base-field reinterpretation of EF rows) is the FFI byte-match
        # slice; until then commit rejects EF up front rather than faulting in the
        # base-field permutation.
        _, _, smcs = _smcs()
        with self.assertRaises(NotImplementedError):
            smcs.commit(jnp.zeros((4, 4), dtype=EF))


class BindStructureTest(absltest.TestCase):
    def test_count_shape_mismatch_raises(self) -> None:
        _, _, smcs = _smcs()
        commitment = jnp.arange(8, dtype=jnp.uint32).view(F)
        with self.assertRaises(ValueError):
            smcs.bind_structure(
                commitment,
                jnp.array([1, 2, 3], dtype=F),
                jnp.array([1, 2], dtype=F),
            )


if __name__ == "__main__":
    absltest.main()
