# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import functools

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.reed_solomon import BitReversedReedSolomon, ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.fri.config import DeepFoldableCode, FriParams
from zorch.pcs.fri.prover import FriProver, _commit_one
from zorch.pcs.fri.verifier import FriVerifier
from zorch.testkit.jit_cache import assert_single_trace
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


class DeepFoldableCodeTest(absltest.TestCase):
    def test_reed_solomon_satisfies_protocol(self) -> None:
        self.assertIsInstance(_params().code, DeepFoldableCode)


class FriCommitCacheTest(absltest.TestCase):
    def test_commit_zone_ignores_open_only_params(self) -> None:
        # commit reads only code/tree, so params differing in the open-side
        # knobs (num_rounds / num_queries) — and freshly built same-config
        # instances — must share one compiled commit zone (#214).
        coeffs = jnp.array([1, 2, 3, 4], dtype=KB)
        calls = [
            functools.partial(
                FriProver(
                    dataclasses.replace(_params(), num_rounds=r, num_queries=q)
                ).commit,
                [coeffs],
            )
            for r, q in ((2, 3), (1, 5))
        ]
        assert_single_trace(self, _commit_one, calls)


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

    def test_malformed_proof_layer_count_raises(self) -> None:
        # A proof missing a fold layer must fail loud: the replay scan iterates
        # over whatever roots it's handed, so a short list would silently skip a
        # round's checks without the eager guard.
        roots, values, proofs = self._prove()
        short = dataclasses.replace(proofs[0], fri_roots=[], query_openings=[])
        with self.assertRaises(ValueError):
            self.verifier.verify(roots, [self.z], values, [short], _transcript())


class FriBitReversedRoundTripTest(absltest.TestCase):
    """FRI over the bit-reversed code: the whole quotient/fold/query path
    must read the pair layout off the seam, including the DEEP quotient's
    domain coordinates at the layer-0 pair."""

    def setUp(self) -> None:
        _, _, tree = koalabear16_merkle()
        code = BitReversedReedSolomon(message_len=4, blowup=2, dtype=KB)
        self.params = FriParams(code=code, tree=tree, num_rounds=2, num_queries=3)
        self.coeffs = jnp.array([1, 2, 3, 4], dtype=KB)
        self.z = jnp.array(2, dtype=KB)
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
