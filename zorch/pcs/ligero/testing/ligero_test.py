# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Single-shot Ligero open/verify: prover↔verifier round-trip on the multiplicative
Reed-Solomon code, against an independent multilinear evaluation (`eval_mle`) of
the committed polynomial. The tamper cases pin the two Ligero checks — proximity
(the opened codeword rows) and value (the sent vector `w`) — plus the Merkle and
commitment-root bindings. Reed-Solomon here is the de-risk vehicle for the
code-generic core; the binary-field (GHASH) instantiation is deferred to the
additive-NTT code (fractalyze/flock-zorch#11, #27).
"""
from __future__ import annotations

import dataclasses

import frx.numpy as jnp
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.ligero.prover import LigeroProver, LigeroProverData
from zorch.pcs.ligero.verifier import LigeroVerifier
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_ext_field
from zorch.transcript import DuplexTranscript


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return rand_ext_field(seed, shape, F, EF)


def _setup(
    log_rows: int, log_cols: int, blowup: int = 2, coset: bool = False
) -> tuple[
    LigeroProver, LigeroVerifier, jnp.ndarray, jnp.ndarray, LigeroProverData, int
]:
    rows = 1 << log_rows
    shift = _rand_ef(99, ()) if coset else None
    code = ReedSolomon(message_len=rows, blowup=blowup, dtype=EF, coset_shift=shift)
    _, _, tree = koalabear16_merkle()
    prover = LigeroProver(code, tree, num_queries=4)
    verifier = LigeroVerifier(code, tree, num_queries=4)
    num_vars = log_rows + log_cols
    f = _rand_ef(1, (1 << num_vars,))
    root, pdata = prover.commit([f])
    return prover, verifier, root, f, pdata, num_vars


class LigeroTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="sq_2x2", log_rows=1, log_cols=1),
        dict(testcase_name="sq_4x4", log_rows=2, log_cols=2),
        dict(testcase_name="wide_2x8", log_rows=1, log_cols=3),
        dict(testcase_name="tall_8x2", log_rows=3, log_cols=1),
        dict(testcase_name="one_col", log_rows=2, log_cols=0),
    )
    def test_open_verify_round_trip(self, log_rows: int, log_cols: int) -> None:
        prover, verifier, root, f, pdata, num_vars = _setup(log_rows, log_cols)
        z = _rand_ef(2, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        # KAT: the opened value is the committed poly's evaluation at z.
        self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_open_verify_round_trip_coset(self) -> None:
        prover, verifier, root, f, pdata, num_vars = _setup(2, 2, coset=True)
        z = _rand_ef(3, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_verify_rejects_tampered_w(self) -> None:
        prover, verifier, root, _f, pdata, num_vars = _setup(2, 2)
        z = _rand_ef(4, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        bad = dataclasses.replace(proof, w=proof.w + jnp.array(1, EF))
        ok, _ = verifier.verify(root, [z], value, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_rejects_tampered_component_opening(self) -> None:
        prover, verifier, root, _f, pdata, num_vars = _setup(2, 2)
        z = _rand_ef(5, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        co = proof.component_opening
        bad_co = dataclasses.replace(co, row=co.row + jnp.array(1, F))
        bad = dataclasses.replace(proof, component_opening=bad_co)
        ok, _ = verifier.verify(root, [z], value, bad, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_rejects_tampered_value(self) -> None:
        prover, verifier, root, _f, pdata, num_vars = _setup(2, 2)
        z = _rand_ef(6, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        ok, _ = verifier.verify(
            root, [z], value + jnp.array(1, EF), proof, _transcript()
        )
        self.assertFalse(bool(ok))

    def test_verify_rejects_wrong_root(self) -> None:
        prover, verifier, root, _f, pdata, num_vars = _setup(2, 2)
        z = _rand_ef(7, (num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        wrong = root + jnp.ones_like(root)
        ok, _ = verifier.verify(wrong, [z], value, proof, _transcript())
        self.assertFalse(bool(ok))

    def test_commit_requires_single_poly(self) -> None:
        prover, _v, _r, f, _pd, _nv = _setup(2, 2)
        with self.assertRaisesRegex(ValueError, "exactly one polynomial"):
            prover.commit([f, f])

    def test_commit_rejects_non_power_of_two(self) -> None:
        # rows=4 (pow2); length 12 = 4*3 is divisible by rows but not a power of
        # two, so the col count 3 is not a valid multilinear axis. Fail loud at
        # commit rather than later at open.
        prover, _v, _r, _f, _pd, _nv = _setup(2, 2)
        with self.assertRaisesRegex(ValueError, "power of two"):
            prover.commit([_rand_ef(8, (12,))])


if __name__ == "__main__":
    absltest.main()
