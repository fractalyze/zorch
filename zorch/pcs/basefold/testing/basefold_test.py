# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BasefoldProver.commit — RS low-degree-extension + Merkle, against an independent
reconstruction of the same codeword + root."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript


def _basefold(
    log_s: int = 2, blowup: int = 2
) -> tuple[BasefoldProver, ReedSolomon, MerkleTree, int]:
    S = 1 << log_s
    rs = ReedSolomon(message_len=S, blowup=blowup, dtype=F)
    _, _, tree = koalabear16_merkle()
    return BasefoldProver(rs, tree), rs, tree, S


def _columns(mle: jnp.ndarray) -> list[jnp.ndarray]:
    """The seam commits a Sequence of column MLEs (1D), like kzg/fri."""
    return [mle[:, j] for j in range(mle.shape[1])]


class BasefoldTest(absltest.TestCase):
    def test_commit_matches_independent_encode_merkle(self) -> None:
        bf, rs, tree, S = _basefold()
        K = 3
        mle = jnp.arange(S * K, dtype=F).reshape(S, K)  # [S, K]
        root, pdata = bf.commit(_columns(mle))
        # Independent reconstruction: column-wise RS-encode then Merkle.
        codeword = rs.encode(mle.T).T  # [S*blowup, K]
        exp_root, _ = tree.commit(codeword)
        self.assertEqual(root.tolist(), exp_root.tolist())
        self.assertIsInstance(pdata, BasefoldProverData)
        self.assertEqual(pdata.widths, (K,))

    def test_prover_data_pytree_round_trips(self) -> None:
        bf, rs, tree, S = _basefold()
        mle = jnp.arange(S * 3, dtype=F).reshape(S, 3)
        _, pdata = bf.commit(_columns(mle))
        leaves, treedef = jax.tree_util.tree_flatten(pdata)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(restored.widths, pdata.widths)
        for a, b in zip(restored.digest_layers, pdata.digest_layers):
            self.assertTrue(bool(jnp.array_equal(a, b)))
        # mle/codeword are data leaves -> must survive the round-trip too.
        self.assertTrue(bool(jnp.array_equal(restored.mle, pdata.mle)))
        self.assertTrue(bool(jnp.array_equal(restored.codeword, pdata.codeword)))

    def test_proof_pytree_round_trips(self) -> None:
        from zorch.commit.merkle import Opening
        from zorch.pcs.basefold.config import BasefoldProof
        from zorch.pcs.fri.config import LayerOpening

        op = Opening(row=jnp.zeros((2, 3), dtype=F), path=[jnp.zeros((2, 8), dtype=F)])
        proof = BasefoldProof(
            univariate_messages=[(jnp.array(1, F), jnp.array(2, F))],
            fri_roots=[jnp.zeros(8, dtype=F)],
            final_poly=jnp.zeros(2, dtype=F),
            component_opening=LayerOpening(op, op),
            query_openings=[LayerOpening(op, op)],
        )
        leaves, treedef = jax.tree_util.tree_flatten(proof)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(len(restored.univariate_messages), 1)
        self.assertEqual(restored.final_poly.shape, (2,))

    def test_commit_retains_mle_and_codeword(self) -> None:
        bf, rs, _tree, S = _basefold()
        K = 3
        mle = jnp.arange(S * K, dtype=F).reshape(S, K)
        _, pdata = bf.commit(_columns(mle))
        self.assertEqual(pdata.mle.shape, (S, K))
        self.assertEqual(pdata.codeword.shape, (rs.block_len, K))
        self.assertEqual(pdata.mle.tolist(), mle.tolist())


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    # EF element = 4 base-field limbs; rand_field emits base fields.
    return rand_field(seed, (*shape, 4), F).view(EF).reshape(shape)


class BasefoldOpenTest(absltest.TestCase):
    def _commit(self, log_s: int, K: int, blowup: int = 2) -> tuple[
        BasefoldProver,
        BasefoldVerifier,
        jnp.ndarray,
        BasefoldProverData,
        jnp.ndarray,
        int,
    ]:
        S = 1 << log_s
        rs = ReedSolomon(message_len=S, blowup=blowup, dtype=EF)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(rs, tree, num_queries=4)
        verifier = BasefoldVerifier(rs, tree, num_queries=4)
        mle = _rand_ef(1, (S, K))  # [S, K]
        cols = [mle[:, k] for k in range(K)]
        root, pdata = prover.commit(cols)
        return prover, verifier, root, pdata, mle, log_s

    def test_open_verify_round_trip_k1(self) -> None:
        prover, verifier, root, pdata, mle, log_s = self._commit(log_s=3, K=1)
        z = _rand_ef(2, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        self.assertEqual(values[0].tolist(), eval_mle(mle[:, 0], z).tolist())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_open_verify_round_trip_multi_column(self) -> None:
        prover, verifier, root, pdata, mle, log_s = self._commit(log_s=3, K=4)
        z = _rand_ef(5, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        for val, col in zip(values, mle.T, strict=True):
            self.assertEqual(val.tolist(), eval_mle(col, z).tolist())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_verify_rejects_tampered_univariate_message(self) -> None:
        prover, verifier, root, pdata, _mle, log_s = self._commit(log_s=3, K=2)
        z = _rand_ef(7, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        z0, o0 = proof.univariate_messages[0]
        bad = dataclasses.replace(
            proof,
            univariate_messages=[
                (z0 + jnp.array(1, EF), o0),
                *proof.univariate_messages[1:],
            ],
        )
        ok, _ = verifier.verify(root, [z], values, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_rejects_tampered_query_opening(self) -> None:
        prover, verifier, root, pdata, _mle, log_s = self._commit(log_s=3, K=2)
        z = _rand_ef(8, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        comp = proof.component_opening
        bad_lo = dataclasses.replace(comp.lo, row=comp.lo.row + jnp.array(1, EF))
        bad = dataclasses.replace(
            proof, component_opening=dataclasses.replace(comp, lo=bad_lo)
        )
        ok, _ = verifier.verify(root, [z], values, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_rejects_wrong_root(self) -> None:
        # Directly exercises the commitment-root binding: verifying a valid proof
        # against a different root must fail (the FS challenges diverge and the
        # Merkle root reconstruction mismatches).
        prover, verifier, root, pdata, _mle, log_s = self._commit(log_s=3, K=2)
        z = _rand_ef(9, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        wrong_root = root + jnp.ones_like(root)
        ok, _ = verifier.verify(wrong_root, [z], values, proof, _transcript())
        self.assertFalse(bool(ok))

    def test_open_compiles_once_per_shape(self) -> None:
        log_s, K = 3, 2  # message_len = 1<<log_s = 8 rows; z has log_s vars
        S = 1 << log_s
        rs = ReedSolomon(message_len=S, blowup=2, dtype=EF)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(rs, tree, num_queries=4)
        t0 = _transcript()  # built eagerly; closed over as a pytree constant

        @functools.partial(jax.jit, static_argnums=())
        def run(mle: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            cols = [mle[:, k] for k in range(K)]
            _, pdata = prover.commit(cols)
            values, proof, _ = prover.open(pdata, [z], t0)
            return values

        for seed in (1, 2, 3):
            run(
                _rand_ef(seed, (S, K)), _rand_ef(seed + 100, (log_s,))
            ).block_until_ready()
        self.assertEqual(run._cache_size(), 1)


if __name__ == "__main__":
    absltest.main()
