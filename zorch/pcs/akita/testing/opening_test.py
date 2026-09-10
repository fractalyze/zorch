# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita opening — round trips against a direct multilinear evaluation, the
packed layout, and one rejection per thing a proof binds.

Every claimed value is checked against `eval_mle` on the committed polynomial,
a route to it that shares nothing with the opening. Each tamper test changes
exactly one input — the commitment, a point, a claimed value, or one part of
the proof — and expects a rejection rather than an exception: a wrong proof is
the verifier's business, and only a malformed one is the caller's.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from hash_frx.sha256 import HostSha256
from lattice_frx.ring import Eval, RnsRing

from zorch.byte_transcript import ByteHashTranscript, ByteTranscript
from zorch.pcs.akita.challenge import FixedWeightTernary
from zorch.pcs.akita.commit import AkitaCommitter
from zorch.pcs.akita.config import AkitaConfig, Decomposition, SisProfile
from zorch.pcs.akita.prover import AkitaProver
from zorch.pcs.akita.verifier import AkitaVerifier
from zorch.pcs.akita.wire import (
    AkitaOpeningClaim,
    AkitaOpeningProof,
    digits_to_field,
    packed_layout,
)
from zorch.pcs.stage import OpeningProof, OpeningWitness
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_field

# The NTT-friendly 36-bit pair `commit_test` uses, at a degree kept small for
# test wall time.
_Q = (34359753217, 34359754753)
_D = 64
_LOG_BASE = 8
_WEIGHT = 8
# One length per layout branch: 512 splits into 2 positions × 4 super-blocks,
# the two 256s into 2 × 2 and can share a point, 64 is exactly one ring
# element, and 32 is shorter than one, so its point never reaches a block
# coordinate.
_LENS = (512, 256, 256, 64, 32)
_FIELD = zk_dtypes.goldilocks


def _config(lens: tuple[int, ...] = _LENS) -> AkitaConfig:
    modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
    profile = SisProfile(_D, _Q)
    return AkitaConfig(
        profile=profile,
        inner_decomposition=Decomposition.covering(_LOG_BASE, modulus // 2),
        outer_decomposition=Decomposition.covering(_LOG_BASE, profile.modulus // 2),
        inner_rows=2,
        outer_rows=2,
        message_lens=lens,
    )


def _uniform_matrix(
    ring: RnsRing, rng: np.random.Generator, rows: int, cols: int
) -> Eval:
    def element() -> Eval:
        host = np.array(
            [rng.integers(0, q, size=ring.d, dtype=np.uint64) for q in ring.q_moduli],
            dtype=np.uint64,
        )
        return ring.eval_from_host(host)

    return ring.stack(
        [ring.stack([element() for _ in range(cols)]) for _ in range(rows)]
    )


def _committer(config: AkitaConfig) -> AkitaCommitter:
    ring = config.profile.ring
    rng = np.random.default_rng(7)
    inner = _uniform_matrix(ring, rng, config.inner_rows, config.inner_cols)
    outer = _uniform_matrix(ring, rng, config.outer_rows, config.outer_cols)
    return AkitaCommitter(config, inner, outer)


def _polys(seed: int) -> list[frx.Array]:
    return [rand_field(seed + i, (length,), _FIELD) for i, length in enumerate(_LENS)]


def _point(seed: int, message: int) -> frx.Array:
    return rand_field(seed, (_LENS[message].bit_length() - 1,), _FIELD)


def _transcript() -> ByteTranscript:
    return ByteHashTranscript.new(b"akita-opening-test", HostSha256())


def _ints(values: frx.Array) -> list[int]:
    return [int(v) for v in np.asarray(values).astype(object).reshape(-1)]


class AkitaOpeningTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.config = _config()
        cls.ring = cls.config.profile.ring
        cls.committer = _committer(cls.config)
        cls.polys = _polys(1)
        cls.commitment, cls.data = cls.committer.commit(cls.polys)
        policy = FixedWeightTernary(_D, _WEIGHT)
        cls.prover = AkitaProver(cls.committer, policy)
        cls.verifier = AkitaVerifier(cls.committer, policy)

    def _open(
        self, messages: list[int], points: list[frx.Array]
    ) -> tuple[AkitaOpeningClaim, OpeningProof[AkitaOpeningProof]]:
        claim = AkitaOpeningClaim(self.commitment, tuple(points), tuple(messages))
        proof, _ = self.prover.open(claim, OpeningWitness(self.data), _transcript())
        return claim, proof

    def _accepts(
        self, claim: AkitaOpeningClaim, proof: OpeningProof[AkitaOpeningProof]
    ) -> bool:
        ok, _ = self.verifier.verify(claim, proof, _transcript())
        return ok

    def _assert_values(
        self, messages: list[int], points: list[frx.Array], values: frx.Array
    ) -> None:
        expected = [
            int(np.asarray(eval_mle(self.polys[message], point)).astype(object))
            for message, point in zip(messages, points)
        ]
        self.assertEqual(_ints(values), expected)

    def test_every_layout_branch_round_trips(self) -> None:
        for message in range(len(_LENS)):
            with self.subTest(message=message):
                point = _point(10 + message, message)
                claim, proof = self._open([message], [point])
                self.assertTrue(self._accepts(claim, proof))
                self._assert_values([message], [point], proof.values)

    def test_packed_claims_share_a_response_per_point(self) -> None:
        shared = _point(20, 1)
        messages = [1, 2, 0, 1]
        points = [shared, shared, _point(21, 0), _point(22, 1)]
        claim, proof = self._open(messages, points)
        self.assertTrue(self._accepts(claim, proof))
        self._assert_values(messages, points, proof.values)
        # Three distinct points, so three responses; the shared point's two
        # members travel as one partial table over one response.
        self.assertLen(proof.proof.responses, 3)
        self.assertEqual(tuple(proof.proof.partials[0].shape), (2, 2, _D))

    def test_layout_splits_each_point_over_its_blocks(self) -> None:
        claim = AkitaOpeningClaim(
            self.commitment,
            tuple(_point(30 + m, m) for m in range(len(_LENS))),
            tuple(range(len(_LENS))),
        )
        self.assertEqual(
            [(g.positions, g.super_blocks) for g in packed_layout(self.config, claim)],
            [(2, 4), (2, 2), (2, 2), (1, 1), (1, 1)],
        )

    def test_weights_factor_the_evaluation(self) -> None:
        # The MSB split, pinned without the prover: the outer product of the
        # three tables is the eq table of the whole point.
        for message in (0, 4):
            with self.subTest(message=message):
                point = _point(40 + message, message)
                claim = AkitaOpeningClaim(self.commitment, (point,), (message,))
                (group,) = packed_layout(self.config, claim)
                inner, position, block = group.weights(_D)
                table = (
                    block[:, None, None]
                    * position[None, :, None]
                    * inner[None, None, :]
                ).reshape(-1)
                full = expand_eq_to_hypercube(point, fnp.ones((), _FIELD))
                self.assertEqual(_ints(table)[: _LENS[message]], _ints(full))

    def test_digits_recompose_to_the_field_coefficients(self) -> None:
        coefficients = digits_to_field(self.ring, self.data.witness, _LOG_BASE, _FIELD)
        self.assertEqual(_ints(coefficients[:8].reshape(-1)), _ints(self.polys[0]))

    def test_prover_and_verifier_leave_the_same_transcript(self) -> None:
        point = _point(50, 0)
        claim = AkitaOpeningClaim(self.commitment, (point,), (0,))
        proof, proved = self.prover.open(
            claim, OpeningWitness(self.data), _transcript()
        )
        ok, verified = self.verifier.verify(claim, proof, _transcript())
        self.assertTrue(ok)
        self.assertEqual(proved.sample_scalar(32)[1], verified.sample_scalar(32)[1])

    def test_rejects_a_different_commitment(self) -> None:
        point = _point(60, 0)
        claim, proof = self._open([0], [point])
        other, _ = self.committer.commit(_polys(9))
        forged = AkitaOpeningClaim(other, claim.points, claim.messages)
        self.assertFalse(self._accepts(forged, proof))

    def test_rejects_a_different_point(self) -> None:
        claim, proof = self._open([0], [_point(61, 0)])
        moved = AkitaOpeningClaim(self.commitment, (_point(62, 0),), claim.messages)
        self.assertFalse(self._accepts(moved, proof))

    def test_rejects_a_wrong_claimed_value(self) -> None:
        claim, proof = self._open([0], [_point(63, 0)])
        wrong = proof.values.at[0].add(fnp.ones((), _FIELD))
        self.assertFalse(self._accepts(claim, OpeningProof(wrong, proof.proof)))

    def test_rejects_tampered_images(self) -> None:
        claim, proof = self._open([0], [_point(64, 0)])
        body = proof.proof
        shifted = self.ring.add(
            body.images, self.ring.from_signed([1] + [0] * (_D - 1))
        )
        tampered = AkitaOpeningProof(shifted, body.partials, body.responses)
        self.assertFalse(self._accepts(claim, OpeningProof(proof.values, tampered)))

    def test_rejects_partials_that_still_sum_to_the_value(self) -> None:
        # Move mass between two super-blocks so `Σ_s B_s·⟨I, E_s⟩` is unchanged:
        # the value check passes and only the fold's evaluation relation can
        # catch the partials lying.
        point = _point(65, 0)
        claim, proof = self._open([0], [point])
        (group,) = packed_layout(self.config, claim)
        _, _, block = group.weights(_D)
        body = proof.proof
        shift = rand_field(66, (_D,), _FIELD)
        partial = body.partials[0]
        partial = partial.at[0, 0].add(block[1] * shift)
        partial = partial.at[0, 1].add(-(block[0] * shift))
        tampered = AkitaOpeningProof(body.images, (partial,), body.responses)
        self.assertFalse(self._accepts(claim, OpeningProof(proof.values, tampered)))

    def test_rejects_a_tampered_response(self) -> None:
        claim, proof = self._open([0], [_point(67, 0)])
        body = proof.proof
        response = self.ring.add(
            body.responses[0], self.ring.from_signed([1] + [0] * (_D - 1))
        )
        tampered = AkitaOpeningProof(body.images, body.partials, (response,))
        self.assertFalse(self._accepts(claim, OpeningProof(proof.values, tampered)))

    def test_rejects_an_over_bound_response(self) -> None:
        claim, proof = self._open([0], [_point(68, 0)])
        body = proof.proof
        spike = self.ring.from_signed([1 << 30] + [0] * (_D - 1))
        response = self.ring.add(body.responses[0], spike)
        tampered = AkitaOpeningProof(body.images, body.partials, (response,))
        self.assertFalse(self._accepts(claim, OpeningProof(proof.values, tampered)))

    def test_refuses_a_proof_missing_a_group(self) -> None:
        claim, proof = self._open([0, 1], [_point(69, 0), _point(70, 1)])
        body = proof.proof
        short = AkitaOpeningProof(body.images, body.partials[:1], body.responses[:1])
        with self.assertRaisesRegex(ValueError, "for 2 groups"):
            self.verifier.verify(
                claim, OpeningProof(proof.values, short), _transcript()
            )

    def test_refuses_a_message_whose_length_is_not_a_power_of_two(self) -> None:
        config = _config((_D + 5,))
        claim = AkitaOpeningClaim(
            self.commitment, (rand_field(71, (7,), _FIELD),), (0,)
        )
        with self.assertRaisesRegex(ValueError, "not a power of two"):
            packed_layout(config, claim)

    def test_refuses_a_point_of_the_wrong_length(self) -> None:
        claim = AkitaOpeningClaim(self.commitment, (_point(72, 1),), (0,))
        with self.assertRaisesRegex(ValueError, "takes a point of 9"):
            packed_layout(self.config, claim)


if __name__ == "__main__":
    absltest.main()
