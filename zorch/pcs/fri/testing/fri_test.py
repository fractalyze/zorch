# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.fri.config import FriParams
from zorch.pcs.fri.prover import FriProver
from zorch.pcs.fri.verifier import FriVerifier
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont  # match the koalabear16 poseidon2 fixture's form


def _params(coset_shift: Array | None = None) -> FriParams:
    # message_len 4 -> block_len 8; quotient (deg 2) folds in 2 rounds to a constant.
    _, _, tree = koalabear16_merkle()
    code = ReedSolomon(message_len=4, blowup=2, dtype=KB, coset_shift=coset_shift)
    return FriParams(code=code, tree=tree, num_rounds=2, num_queries=3)


def _transcript() -> DuplexTranscript:
    # A cheap deterministic sponge; prover and verifier each build a fresh,
    # identical one, so both draw the same Fiat-Shamir stream (fold challenges
    # then query positions).
    return cheap_transcript(KB)


class FriRoundTripTest(absltest.TestCase):
    def setUp(self) -> None:
        self.params = _params()
        self.coeffs = jnp.array([1, 2, 3, 4], dtype=KB)
        self.z = jnp.array(2, dtype=KB)  # outside the order-8 subgroup
        self.prover = FriProver(self.params)
        self.verifier = FriVerifier(self.params)

    def _prove(self) -> tuple[Array, Array, list]:
        roots, data = self.prover.commit([self.coeffs])
        values, proofs, _ = self.prover.open(data, [self.z], _transcript())
        return roots, values, proofs

    def test_honest_opening_verifies(self) -> None:
        roots, values, proofs = self._prove()
        ok, _ = self.verifier.verify(roots, [self.z], values, proofs, _transcript())
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        roots, values, proofs = self._prove()
        bad = values + jnp.array(1, dtype=KB)
        ok, _ = self.verifier.verify(roots, [self.z], bad, proofs, _transcript())
        self.assertFalse(bool(ok))

    def test_tampered_final_layer_rejected(self) -> None:
        roots, values, proofs = self._prove()
        pf = proofs[0]
        tampered = dataclasses.replace(
            pf, final_layer=pf.final_layer.at[0].add(jnp.array(1, dtype=KB))
        )
        ok, _ = self.verifier.verify(roots, [self.z], values, [tampered], _transcript())
        self.assertFalse(bool(ok))

    def test_batch_length_mismatch_raises(self) -> None:
        # A short batch must fail loud, not silently truncate to the common prefix.
        roots, data = self.prover.commit([self.coeffs])
        with self.assertRaises(ValueError):
            self.prover.open(data, [self.z, self.z], _transcript())
        with self.assertRaises(ValueError):
            self.verifier.verify(
                roots, [self.z, self.z], jnp.zeros(2, dtype=KB), [], _transcript()
            )


class FriCosetRoundTripTest(absltest.TestCase):
    """FRI on a coset LDE (the STARK shape: eval domain disjoint from the trace
    domain). Every layer folds over the shifted domain, squaring the shift."""

    def setUp(self) -> None:
        self.params = _params(coset_shift=jnp.array(3, dtype=KB))
        self.coeffs = jnp.array([1, 2, 3, 4], dtype=KB)
        self.z = jnp.array(2, dtype=KB)  # outside the coset 3·<ω₈>
        self.prover = FriProver(self.params)
        self.verifier = FriVerifier(self.params)

    def test_honest_opening_verifies(self) -> None:
        roots, data = self.prover.commit([self.coeffs])
        values, proofs, _ = self.prover.open(data, [self.z], _transcript())
        ok, _ = self.verifier.verify(roots, [self.z], values, proofs, _transcript())
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        roots, data = self.prover.commit([self.coeffs])
        values, proofs, _ = self.prover.open(data, [self.z], _transcript())
        bad = values + jnp.array(1, dtype=KB)
        ok, _ = self.verifier.verify(roots, [self.z], bad, proofs, _transcript())
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
