# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito recursive open/verify: prover<->verifier round-trip on the
multiplicative Reed-Solomon code, against an independent multilinear evaluation
(`eval_mle`) of the committed polynomial. Reed-Solomon is the de-risk vehicle for
the code-generic recursion; the binary-field (GHASH) instantiation is deferred to
the additive-NTT code (fractalyze/flock-zorch#11, #27).
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import LigeritoProver, LigeritoProverData
from zorch.pcs.ligerito.verifier import LigeritoVerifier
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_ext_field
from zorch.transcript import DuplexTranscript


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return rand_ext_field(seed, shape, F, EF)


def _make_code(message_len: int, log_inv_rate: int) -> ReedSolomon:
    return ReedSolomon(message_len=message_len, blowup=1 << log_inv_rate, dtype=EF)


def _setup(
    cfg: LigeritoConfig,
) -> tuple[
    LigeritoProver,
    LigeritoVerifier,
    LigeritoCommitment,
    jnp.ndarray,
    LigeritoProverData,
]:
    _, _, tree = koalabear16_merkle()
    prover = LigeritoProver(_make_code, tree, cfg)
    verifier = LigeritoVerifier(_make_code, tree, cfg)
    f = _rand_ef(1, (1 << cfg.num_vars,))
    root, pdata = prover.commit([f])
    return prover, verifier, root, f, pdata


class LigeritoTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # One recursive level (L=2): the smallest case exercising induce/glue.
        dict(
            testcase_name="l2_4v",
            num_vars=4,
            fold_ks=(1, 1),
            rates=(1, 1),
            queries=(4, 4),
        ),
        # Two recursive levels (L=3), residual 3.
        dict(
            testcase_name="l3_6v",
            num_vars=6,
            fold_ks=(1, 1, 1),
            rates=(1, 1, 1),
            queries=(4, 4, 4),
        ),
        # Multi-variable folds per level.
        dict(
            testcase_name="l2_bigfold",
            num_vars=5,
            fold_ks=(2, 1),
            rates=(1, 1),
            queries=(4, 4),
        ),
        # Shrinking rate across levels (the real Ligerito rate schedule).
        dict(
            testcase_name="l2_shrinkrate",
            num_vars=4,
            fold_ks=(1, 1),
            rates=(2, 1),
            queries=(4, 4),
        ),
        # L=1 (no recursion): value + direct proximity + terminal only, no glue.
        dict(
            testcase_name="l1_2res", num_vars=3, fold_ks=(1,), rates=(1,), queries=(4,)
        ),
        # Minimal residual (1 var) across a recursive level.
        dict(
            testcase_name="l2_res1",
            num_vars=3,
            fold_ks=(1, 1),
            rates=(1, 1),
            queries=(2, 2),
        ),
        # Three recursive levels, mixed folds + shrinking rate.
        dict(
            testcase_name="l4_8v",
            num_vars=8,
            fold_ks=(2, 1, 1, 1),
            rates=(3, 2, 1, 1),
            queries=(6, 4, 4, 2),
        ),
    )
    def test_open_verify_round_trip(
        self,
        num_vars: int,
        fold_ks: tuple[int, ...],
        rates: tuple[int, ...],
        queries: tuple[int, ...],
    ) -> None:
        cfg = LigeritoConfig(
            num_vars=num_vars, fold_ks=fold_ks, log_inv_rates=rates, queries=queries
        )
        prover, verifier, root, f, pdata = _setup(cfg)
        z = _rand_ef(2, (cfg.num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))


# A recursive config (L=2) with a residual — exercises every proof component the
# tamper cases below poke: recursive root, opened rows, residual, sumcheck msg.
_TAMPER_CFG = LigeritoConfig(
    num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=(4, 4)
)


class LigeritoTamperTest(parameterized.TestCase):
    def _open(
        self,
    ) -> tuple[
        LigeritoVerifier, LigeritoCommitment, jnp.ndarray, jnp.ndarray, LigeritoProof
    ]:
        prover, verifier, root, f, pdata = _setup(_TAMPER_CFG)
        z = _rand_ef(3, (_TAMPER_CFG.num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        return verifier, root, z, value, proof

    def _reject(
        self,
        verifier: LigeritoVerifier,
        root: LigeritoCommitment,
        z: jnp.ndarray,
        value: jnp.ndarray,
        proof: LigeritoProof,
    ) -> None:
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertFalse(bool(ok))

    def test_rejects_tampered_recursive_root(self) -> None:
        verifier, root, z, value, proof = self._open()
        roots = list(proof.recursive_roots)
        roots[0] = roots[0] + jnp.ones_like(roots[0])
        self._reject(
            verifier, root, z, value, dataclasses.replace(proof, recursive_roots=roots)
        )

    def test_rejects_tampered_opened_row(self) -> None:
        verifier, root, z, value, proof = self._open()
        opens = list(proof.component_openings)
        co = opens[0]
        opens[0] = dataclasses.replace(co, row=co.row + jnp.array(1, F))
        self._reject(
            verifier,
            root,
            z,
            value,
            dataclasses.replace(proof, component_openings=opens),
        )

    def test_rejects_tampered_residual(self) -> None:
        verifier, root, z, value, proof = self._open()
        bad = dataclasses.replace(
            proof, final_residual=proof.final_residual + jnp.array(1, EF)
        )
        self._reject(verifier, root, z, value, bad)

    def test_rejects_tampered_sumcheck_message(self) -> None:
        verifier, root, z, value, proof = self._open()
        msgs = list(proof.sumcheck_messages)
        msgs[0] = msgs[0] + jnp.array(1, EF)
        self._reject(
            verifier, root, z, value, dataclasses.replace(proof, sumcheck_messages=msgs)
        )

    def test_rejects_tampered_value(self) -> None:
        verifier, root, z, value, proof = self._open()
        self._reject(verifier, root, z, value + jnp.array(1, EF), proof)


class LigeritoConfigTest(absltest.TestCase):
    def test_rejects_empty_fold_ks(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            LigeritoConfig(num_vars=4, fold_ks=(), log_inv_rates=(), queries=())

    def test_rejects_non_positive_fold_ks(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            LigeritoConfig(
                num_vars=4, fold_ks=(0, 1), log_inv_rates=(1, 1), queries=(4, 4)
            )


class LigeritoCommitTest(absltest.TestCase):
    def test_commit_independent_of_query_count(self) -> None:
        # Two provers differing ONLY in the query count must produce an identical
        # commitment — commit never reads `queries`, so the jitted `_commit`
        # (keyed on code+tree+interleave, #214) does not re-trace across them.
        def _cfg(queries: tuple[int, ...]) -> LigeritoConfig:
            return LigeritoConfig(
                num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=queries
            )

        cfg_a = _cfg((4, 4))
        cfg_b = _cfg((8, 2))
        _, _, tree = koalabear16_merkle()
        f = _rand_ef(1, (1 << 4,))
        root_a, _ = LigeritoProver(_make_code, tree, cfg_a).commit([f])
        root_b, _ = LigeritoProver(_make_code, tree, cfg_b).commit([f])
        self.assertEqual(root_a.tolist(), root_b.tolist())

    def test_commit_rejects_multiple_polys(self) -> None:
        _, _, tree = koalabear16_merkle()
        f = _rand_ef(1, (1 << 4,))
        prover = LigeritoProver(_make_code, tree, _TAMPER_CFG)
        with self.assertRaisesRegex(ValueError, "exactly one polynomial"):
            prover.commit([f, f])


if __name__ == "__main__":
    absltest.main()
