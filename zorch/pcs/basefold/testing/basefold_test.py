# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BasefoldProver.commit — RS low-degree-extension + Merkle, against an independent
reconstruction of the same codeword + root."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.prover import (
    BasefoldProver,
    BasefoldProverData,
    _commit_body,
    _open_batch_body,
)
from zorch.pcs.basefold.verifier import BasefoldVerifier, _verify_batch_body
from zorch.poly.multilinear import eval_mle
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_ext_field
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

        # num_vars = 1: one fold round (one fri root, one pair-leaf opening), one
        # committed matrix of width 3 opened at the query positions.
        comp = Opening(
            row=jnp.zeros((2, 3), dtype=F), path=[jnp.zeros((2, 8), dtype=F)]
        )
        pair = Opening(
            row=jnp.zeros((2, 2), dtype=F), path=[jnp.zeros((2, 8), dtype=F)]
        )
        proof = BasefoldProof(
            univariate_messages=[(jnp.array(1, F), jnp.array(2, F))],
            fri_roots=[jnp.zeros(8, dtype=F)],
            final_poly=jnp.zeros(2, dtype=F),
            component_openings=[comp],
            query_openings=[pair],
        )
        leaves, treedef = jax.tree_util.tree_flatten(proof)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(len(restored.univariate_messages), 1)
        self.assertEqual(restored.final_poly.shape, (2,))
        self.assertEqual(len(restored.component_openings), 1)

    def test_commit_eager_reuses_one_compiled_zone_across_instances(self) -> None:
        # Standalone (no enclosing jit), `commit` must reuse one compiled zone
        # across calls, across freshly built same-config provers, and across
        # provers differing only in num_queries (which commit never reads) —
        # the zone is module-level, keyed on (code, tree) by value (#214).
        calls: list[Callable[[], object]] = []
        for num_queries in (4, 8):
            bf, _rs, _tree, S = _basefold()
            bf = dataclasses.replace(bf, num_queries=num_queries)
            for offset in (0, 1, 2):
                mle = jnp.arange(S * 2, dtype=F).reshape(S, 2) + F(offset)
                calls.append(functools.partial(bf.commit, _columns(mle)))
        assert_single_trace(self, _commit_body, calls)

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
    return rand_ext_field(seed, shape, F, EF)


class BasefoldOpenTest(absltest.TestCase):
    def _commit(
        self,
        log_s: int,
        K: int,
        blowup: int = 2,
        code_cls: type = ReedSolomon,
    ) -> tuple[
        BasefoldProver,
        BasefoldVerifier,
        jnp.ndarray,
        BasefoldProverData,
        jnp.ndarray,
        int,
    ]:
        S = 1 << log_s
        code = code_cls(message_len=S, blowup=blowup, dtype=EF)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(code, tree, num_queries=4)
        verifier = BasefoldVerifier(code, tree, num_queries=4)
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
        comp = proof.component_openings[0]
        bad_comp = dataclasses.replace(comp, row=comp.row + jnp.array(1, EF))
        bad = dataclasses.replace(
            proof, component_openings=[bad_comp, *proof.component_openings[1:]]
        )
        ok, _ = verifier.verify(root, [z], values, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_open_verify_round_trip_bit_reversed_code(self) -> None:
        # The open/verify path reads the pair layout off the code seam, so
        # swapping in the bit-reversed code must round-trip with no other
        # change — this exercises the adjacent-pair query indexing end to end.
        prover, verifier, root, pdata, mle, log_s = self._commit(
            log_s=3, K=2, code_cls=BitReversedReedSolomon
        )
        z = _rand_ef(13, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        for val, col in zip(values, mle.T, strict=True):
            self.assertEqual(val.tolist(), eval_mle(col, z).tolist())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_verify_rejects_tampered_opening_bit_reversed_code(self) -> None:
        # The adjacent-pair indexing is exactly where a layout bug would hide
        # a forged leaf, so the tamper rejection is re-checked on this code.
        prover, verifier, root, pdata, _mle, log_s = self._commit(
            log_s=3, K=2, code_cls=BitReversedReedSolomon
        )
        z = _rand_ef(15, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        comp = proof.component_openings[0]
        bad_comp = dataclasses.replace(comp, row=comp.row + jnp.array(1, EF))
        bad = dataclasses.replace(
            proof, component_openings=[bad_comp, *proof.component_openings[1:]]
        )
        ok, _ = verifier.verify(root, [z], values, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_open_verify_round_trip_coset(self) -> None:
        # Coset LDE: prover and verifier must thread rs.coset_shift through
        # every fold (squaring it per round); an EF shift exercises the
        # EF-domain path end to end.
        log_s, K = 3, 2
        S = 1 << log_s
        shift = _rand_ef(10, ())
        rs = ReedSolomon(message_len=S, blowup=2, dtype=EF, coset_shift=shift)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(rs, tree, num_queries=4)
        verifier = BasefoldVerifier(rs, tree, num_queries=4)
        mle = _rand_ef(11, (S, K))
        root, pdata = prover.commit([mle[:, k] for k in range(K)])
        z = _rand_ef(12, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_open_rejects_empty_point(self) -> None:
        # num_vars = 0 would skip every fold round and leave nothing to observe;
        # fail loud before the (also-wrong) MLE-height check can mask the cause.
        prover, _, _, pdata, _, _ = self._commit(log_s=3, K=1)
        with self.assertRaisesRegex(ValueError, "at least one variable"):
            prover.open(pdata, [_rand_ef(3, (0,))], _transcript())

    def test_verify_rejects_empty_point(self) -> None:
        # The verifier mirrors the prover's guard: with zero variables the fold
        # replay would index an empty layer list inside the jit zone.
        prover, verifier, root, pdata, _mle, log_s = self._commit(log_s=3, K=1)
        z = _rand_ef(3, (log_s,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        with self.assertRaisesRegex(ValueError, "at least one variable"):
            verifier.verify(root, [_rand_ef(3, (0,))], values, proof, _transcript())

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

    def _batch_commit(
        self, log_s: int, widths: tuple[int, ...], code_cls: type = ReedSolomon
    ) -> tuple[BasefoldProver, BasefoldVerifier, list, list, list, int]:
        S = 1 << log_s
        code = code_cls(message_len=S, blowup=2, dtype=EF)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(code, tree, num_queries=4)
        verifier = BasefoldVerifier(code, tree, num_queries=4)
        roots, pdatas, mles = [], [], []
        for r, w in enumerate(widths):
            mle = _rand_ef(20 + r, (S, w))
            root, pdata = prover.commit([mle[:, k] for k in range(w)])
            roots.append(root)
            pdatas.append(pdata)
            mles.append(mle)
        return prover, verifier, roots, pdatas, mles, log_s

    def test_batch_open_verify_two_matrices_distinct_widths(self) -> None:
        # The #169 deliverable: two separately committed matrices of different
        # widths, reduced to one FRI by the staggered RLC, round-trip.
        prover, verifier, roots, pdatas, mles, log_s = self._batch_commit(
            log_s=3, widths=(2, 3)
        )
        z = _rand_ef(30, (log_s,))
        values, proof, _ = prover.open_batch(pdatas, [z], _transcript())
        for vals, mle in zip(values, mles, strict=True):
            self.assertEqual(vals.tolist(), eval_mle(mle, z, axis=0).tolist())
        ok, _ = verifier.verify_batch(roots, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_batch_open_verify_bit_reversed_code(self) -> None:
        prover, verifier, roots, pdatas, _mles, log_s = self._batch_commit(
            log_s=3, widths=(1, 2, 1), code_cls=BitReversedReedSolomon
        )
        z = _rand_ef(31, (log_s,))
        values, proof, _ = prover.open_batch(pdatas, [z], _transcript())
        ok, _ = verifier.verify_batch(roots, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_batch_verify_rejects_wrong_round_root(self) -> None:
        prover, verifier, roots, pdatas, _mles, log_s = self._batch_commit(
            log_s=3, widths=(2, 3)
        )
        z = _rand_ef(32, (log_s,))
        values, proof, _ = prover.open_batch(pdatas, [z], _transcript())
        bad_roots = [roots[0] + jnp.ones_like(roots[0]), roots[1]]
        ok, _ = verifier.verify_batch(bad_roots, [z], values, proof, _transcript())
        self.assertFalse(bool(ok))

    def test_batch_verify_rejects_tampered_component_opening(self) -> None:
        prover, verifier, roots, pdatas, _mles, log_s = self._batch_commit(
            log_s=3, widths=(2, 3)
        )
        z = _rand_ef(33, (log_s,))
        values, proof, _ = prover.open_batch(pdatas, [z], _transcript())
        co = proof.component_openings[1]
        bad_co = dataclasses.replace(co, row=co.row + jnp.array(1, EF))
        bad = dataclasses.replace(
            proof, component_openings=[proof.component_openings[0], bad_co]
        )
        ok, _ = verifier.verify_batch(roots, [z], values, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_replay_lowers_to_scan(self) -> None:
        # The interleaved sumcheck + fold-challenge replay rides one lax.scan (a
        # stablehlo.while), so the verifier traces the round body once instead of
        # once per fold layer (#185). A regression that unrolls the replay drops
        # the while.
        prover, verifier, root, pdata, _mle, _ = self._commit(log_s=4, K=2)
        z = _rand_ef(44, (4,))
        values, proof, _ = prover.open(pdata, [z], _transcript())
        text = _verify_batch_body.lower(
            verifier, [root], z, [values], proof, _transcript()
        ).as_text()
        self.assertIn("stablehlo.while", text)

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

    def test_open_eager_reuses_one_compiled_zone_across_instances(self) -> None:
        # Standalone (no enclosing jit), `open` must reuse one compiled zone
        # across calls (#186) AND across freshly built same-config provers —
        # the zone is module-level and its static key compares the prover by
        # value (#214), so per-test instances stop re-tracing. Each call also
        # passes a freshly built transcript (the #177 value-equality contract).
        calls: list[Callable[[], object]] = []
        for _ in (0, 1):
            prover, _verifier, _root, pdata, _mle, log_s = self._commit(log_s=3, K=2)
            for seed in (2, 3, 4):
                calls.append(
                    functools.partial(
                        prover.open, pdata, [_rand_ef(seed, (log_s,))], _transcript()
                    )
                )
        assert_single_trace(self, _open_batch_body, calls)


if __name__ == "__main__":
    absltest.main()
