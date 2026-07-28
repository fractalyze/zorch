# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import functools

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.reed_solomon import BitReversedReedSolomon, ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.deep_fri.config import DeepFoldableCode, DeepFriParams, DeepFriProof
from zorch.pcs.deep_fri.prover import DeepFriProver, _commit_one
from zorch.pcs.deep_fri.verifier import DeepFriVerifier
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont  # match the koalabear16 poseidon2 fixture's form


def _params(coset_shift: Array | None = None) -> DeepFriParams:
    # message_len 4 -> block_len 8; the composition (deg 2) folds in 2 rounds
    # to a constant.
    _, _, tree = koalabear16_merkle()
    code = ReedSolomon(message_len=4, blowup=2, dtype=KB, coset_shift=coset_shift)
    return DeepFriParams(code=code, tree=tree, num_rounds=2, num_queries=3)


def _transcript() -> DuplexTranscript:
    # A cheap deterministic sponge; prover and verifier each build a fresh,
    # identical one, so both draw the same Fiat-Shamir stream (batching
    # challenge, then fold challenges, then query positions).
    return cheap_transcript(KB)


class DeepFoldableCodeTest(absltest.TestCase):
    def test_reed_solomon_satisfies_protocol(self) -> None:
        self.assertIsInstance(_params().code, DeepFoldableCode)


class DeepFriParamsTest(absltest.TestCase):
    def test_degenerate_schedules_rejected(self) -> None:
        # Zero rounds satisfies the verifier's structural guard then crashes
        # its layer-0 rebuild; zero queries verifies anything; over-folding
        # runs past the code's depth. All three fail at construction.
        good = _params()
        for rounds, queries in ((0, 3), (4, 3), (2, 0)):
            with self.assertRaises(ValueError):
                dataclasses.replace(good, num_rounds=rounds, num_queries=queries)

    def test_full_depth_schedule_accepted(self) -> None:
        # block_len 8 folds at most 3 times (to one element); the boundary is
        # a valid schedule, not a degenerate one.
        params = dataclasses.replace(_params(), num_rounds=3)
        self.assertEqual(params.num_rounds, 3)


class DeepFriCommitCacheTest(absltest.TestCase):
    def test_commit_zone_ignores_open_only_params(self) -> None:
        # commit reads only code/tree, so params differing in the open-side
        # knobs (num_rounds / num_queries) — and freshly built same-config
        # instances — must share one compiled commit zone (#214).
        coeffs = fnp.array([1, 2, 3, 4], dtype=KB)
        calls = [
            functools.partial(
                DeepFriProver(
                    dataclasses.replace(_params(), num_rounds=r, num_queries=q)
                ).commit,
                [coeffs],
            )
            for r, q in ((2, 3), (1, 5))
        ]
        assert_single_trace(self, _commit_one, calls)


class DeepFriRoundTripTest(absltest.TestCase):
    """The batched opening: three committed polynomials, each at its own
    point, covered by ONE composition and ONE fold chain."""

    def setUp(self) -> None:
        self.params = _params()
        self.polys = [
            fnp.array([1, 2, 3, 4], dtype=KB),
            fnp.array([5, 6, 7, 8], dtype=KB),
            fnp.array([9, 10, 11, 12], dtype=KB),
        ]
        # Outside the order-8 subgroup, pairwise distinct.
        self.points = [fnp.array(z, dtype=KB) for z in (2, 11, 13)]
        self.prover = DeepFriProver(self.params)
        self.verifier = DeepFriVerifier(self.params)

    def _prove(self) -> tuple[Array, Array, DeepFriProof]:
        roots, data = self.prover.commit(self.polys)
        values, proof, _ = self.prover._open(data, self.points, _transcript())
        return roots, values, proof

    def test_honest_opening_verifies(self) -> None:
        roots, values, proof = self._prove()
        ok, _ = self.verifier._verify_opening(
            roots, self.points, values, proof, _transcript()
        )
        self.assertTrue(bool(ok))

    def test_single_fold_chain(self) -> None:
        # The batch shares one composition: num_rounds fold layers total, one
        # f opening per committed poly — not a chain per poly.
        _, _, proof = self._prove()
        self.assertLen(proof.fri_roots, self.params.num_rounds)
        self.assertLen(proof.query_openings, self.params.num_rounds)
        self.assertLen(proof.f_openings, len(self.polys))

    def test_wrong_value_rejected(self) -> None:
        roots, values, proof = self._prove()
        bad = values.at[1].add(fnp.array(1, dtype=KB))
        ok, _ = self.verifier._verify_opening(
            roots, self.points, bad, proof, _transcript()
        )
        self.assertFalse(bool(ok))

    def test_swapped_values_rejected(self) -> None:
        # Batch-integrity: each value must bind to ITS column's quotient, so a
        # cross-column permutation of otherwise-committed values must fail.
        roots, values, proof = self._prove()
        swapped = values[fnp.array([1, 0, 2])]
        ok, _ = self.verifier._verify_opening(
            roots, self.points, swapped, proof, _transcript()
        )
        self.assertFalse(bool(ok))

    def test_tampered_final_layer_rejected(self) -> None:
        roots, values, proof = self._prove()
        tampered = dataclasses.replace(
            proof, final_layer=proof.final_layer.at[0].add(fnp.array(1, dtype=KB))
        )
        ok, _ = self.verifier._verify_opening(
            roots, self.points, values, tampered, _transcript()
        )
        self.assertFalse(bool(ok))

    def test_batch_length_mismatch_raises(self) -> None:
        # A short batch must fail loud, not silently truncate to the common prefix.
        _, data = self.prover.commit(self.polys)
        with self.assertRaises(ValueError):
            self.prover._open(data, self.points[:2], _transcript())
        roots, values, proof = self._prove()
        with self.assertRaises(ValueError):
            self.verifier._verify_opening(
                roots, self.points[:2], values, proof, _transcript()
            )

    def test_malformed_proof_layer_count_raises(self) -> None:
        # A proof missing a fold layer must fail loud: the replay scan iterates
        # over whatever roots it's handed, so a short list would silently skip a
        # round's checks without the eager guard.
        roots, values, proof = self._prove()
        short = dataclasses.replace(proof, fri_roots=[], query_openings=[])
        with self.assertRaises(ValueError):
            self.verifier._verify_opening(
                roots, self.points, values, short, _transcript()
            )


class DeepFriSinglePolyRoundTripTest(absltest.TestCase):
    """M = 1 is the batch's degenerate case, not a separate code path."""

    def setUp(self) -> None:
        self.params = _params()
        self.coeffs = fnp.array([1, 2, 3, 4], dtype=KB)
        self.z = fnp.array(2, dtype=KB)  # outside the order-8 subgroup
        self.prover = DeepFriProver(self.params)
        self.verifier = DeepFriVerifier(self.params)

    def test_honest_opening_verifies(self) -> None:
        roots, data = self.prover.commit([self.coeffs])
        values, proof, _ = self.prover._open(data, [self.z], _transcript())
        ok, _ = self.verifier._verify_opening(
            roots, [self.z], values, proof, _transcript()
        )
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        roots, data = self.prover.commit([self.coeffs])
        values, proof, _ = self.prover._open(data, [self.z], _transcript())
        bad = values + fnp.array(1, dtype=KB)
        ok, _ = self.verifier._verify_opening(
            roots, [self.z], bad, proof, _transcript()
        )
        self.assertFalse(bool(ok))


class DeepFriBitReversedRoundTripTest(absltest.TestCase):
    """DEEP-FRI over the bit-reversed code: the whole composition/fold/query
    path must read the pair layout off the seam, including the composition's
    domain coordinates at the layer-0 pair."""

    def setUp(self) -> None:
        _, _, tree = koalabear16_merkle()
        code = BitReversedReedSolomon(message_len=4, blowup=2, dtype=KB)
        self.params = DeepFriParams(code=code, tree=tree, num_rounds=2, num_queries=3)
        self.polys = [
            fnp.array([1, 2, 3, 4], dtype=KB),
            fnp.array([5, 6, 7, 8], dtype=KB),
        ]
        self.points = [fnp.array(2, dtype=KB), fnp.array(11, dtype=KB)]
        self.prover = DeepFriProver(self.params)
        self.verifier = DeepFriVerifier(self.params)

    def test_honest_opening_verifies(self) -> None:
        roots, data = self.prover.commit(self.polys)
        values, proof, _ = self.prover._open(data, self.points, _transcript())
        ok, _ = self.verifier._verify_opening(
            roots, self.points, values, proof, _transcript()
        )
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        roots, data = self.prover.commit(self.polys)
        values, proof, _ = self.prover._open(data, self.points, _transcript())
        bad = values.at[0].add(fnp.array(1, dtype=KB))
        ok, _ = self.verifier._verify_opening(
            roots, self.points, bad, proof, _transcript()
        )
        self.assertFalse(bool(ok))


class DeepFriCosetRoundTripTest(absltest.TestCase):
    """DEEP-FRI on a coset LDE (the STARK shape: eval domain disjoint from the
    trace domain). Every layer folds over the shifted domain, squaring the
    shift."""

    def setUp(self) -> None:
        self.params = _params(coset_shift=fnp.array(3, dtype=KB))
        self.polys = [
            fnp.array([1, 2, 3, 4], dtype=KB),
            fnp.array([5, 6, 7, 8], dtype=KB),
        ]
        # Outside the coset 3·<ω₈>.
        self.points = [fnp.array(2, dtype=KB), fnp.array(11, dtype=KB)]
        self.prover = DeepFriProver(self.params)
        self.verifier = DeepFriVerifier(self.params)

    def test_honest_opening_verifies(self) -> None:
        roots, data = self.prover.commit(self.polys)
        values, proof, _ = self.prover._open(data, self.points, _transcript())
        ok, _ = self.verifier._verify_opening(
            roots, self.points, values, proof, _transcript()
        )
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        roots, data = self.prover.commit(self.polys)
        values, proof, _ = self.prover._open(data, self.points, _transcript())
        bad = values.at[0].add(fnp.array(1, dtype=KB))
        ok, _ = self.verifier._verify_opening(
            roots, self.points, bad, proof, _transcript()
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
