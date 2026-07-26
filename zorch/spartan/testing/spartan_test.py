# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.pcs.stage import OpeningClaim, OpeningProof
from zorch.spartan.lincheck import InnerProver, InnerVerifier
from zorch.spartan.pcs_glue import WitnessOpenProver, WitnessOpenVerifier
from zorch.spartan.r1cs import R1CS
from zorch.spartan.spartan import (
    SpartanClaim,
    SpartanProof,
    SpartanProver,
    SpartanVerifier,
    SpartanWitness,
    _absorb_claim,
    _observe_framed,
)
from zorch.spartan.testing.dense_pcs import DensePcs
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import OuterProver, OuterVerifier
from zorch.stage import ProverStage, TrivialClaim, VerifierStage
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont

# The transcript's own field: the schedule these tests pinned before the
# policy required an explicit field.
_CH = ChallengePolicy(KB)
KBX4 = zk_dtypes.koalabearx4_mont


class _VerifierOnlyPcs(
    VerifierStage[OpeningClaim[Array], TrivialClaim, OpeningProof[Array]]
):
    """Expose no commit/open capability to the deployed verifier."""

    def __init__(self, pcs: DensePcs) -> None:
        self._pcs = pcs

    def verify(self, *args: Any) -> Any:
        return self._pcs.verify(*args)


def _prove_verify(
    seed: int, s_x: int, nvp: int, num_io: int
) -> tuple[R1CS, Array, SpartanProof, _VerifierOnlyPcs, Array]:
    instance, z, _, io = toy_r1cs(
        seed, s_x=s_x, num_vars_padded=nvp, num_io=num_io, dtype=KB
    )
    pcs = DensePcs()
    verifier_pcs = _VerifierOnlyPcs(pcs)
    claim = SpartanClaim(instance, io)
    proved = SpartanProver(pcs, challenges=_CH).prove(
        claim, SpartanWitness(z), cheap_transcript(KB)
    )
    verified = SpartanVerifier(verifier_pcs, challenges=_CH).verify(
        claim, proved.reduction_proof, cheap_transcript(KB)
    )
    return instance, io, proved.reduction_proof, verifier_pcs, verified.ok


class _RecordingProver(ProverStage[Any, Any, Any, Any]):
    def __init__(self, stage: ProverStage[Any, Any, Any, Any]) -> None:
        self.stage = stage
        self.transcript: Transcript | None = None
        self.claim: Any = None

    def prove(self, *args: Any) -> Any:
        result = self.stage.prove(*args)
        self.transcript = result.transcript
        self.claim = result.reduced_claim
        return result


class _RecordingVerifier(VerifierStage[Any, Any, Any]):
    def __init__(self, stage: VerifierStage[Any, Any, Any]) -> None:
        self.stage = stage
        self.transcript: Transcript | None = None
        self.claim: Any = None

    def verify(self, *args: Any) -> Any:
        result = self.stage.verify(*args)
        self.transcript = result.transcript
        self.claim = result.reduced_claim
        return result


def _assert_boundary_agrees(
    case: absltest.TestCase,
    prover: _RecordingProver,
    verifier: _RecordingVerifier,
) -> None:
    assert prover.transcript is not None
    assert verifier.transcript is not None
    _, prover_next = prover.transcript.sample(1)
    _, verifier_next = verifier.transcript.sample(1)
    case.assertTrue(bool(fnp.all(prover_next == verifier_next)))
    if prover.claim is not None:
        for field in prover.claim.__dataclass_fields__:
            case.assertTrue(
                bool(
                    fnp.all(
                        getattr(prover.claim, field) == getattr(verifier.claim, field)
                    )
                )
            )


class SpartanE2ETest(absltest.TestCase):
    def test_roundtrip_accepts_with_verifier_only_pcs(self) -> None:
        _, _, _, verifier_pcs, ok = _prove_verify(1, 2, 4, 2)
        self.assertTrue(bool(ok))
        self.assertFalse(hasattr(verifier_pcs, "commit"))
        self.assertFalse(hasattr(verifier_pcs, "open"))

    def test_composite_transcripts_agree_per_boundary(self) -> None:
        instance, z, _, io = toy_r1cs(2, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        outer_p = _RecordingProver(OuterProver(challenges=_CH))
        outer_v = _RecordingVerifier(OuterVerifier(challenges=_CH))
        inner_p = _RecordingProver(InnerProver(challenges=_CH))
        inner_v = _RecordingVerifier(InnerVerifier(challenges=_CH))
        opening_p = _RecordingProver(WitnessOpenProver(pcs))
        opening_v = _RecordingVerifier(WitnessOpenVerifier(_VerifierOnlyPcs(pcs)))
        prover = SpartanProver(
            pcs, outer=outer_p, inner=inner_p, witness_open=opening_p, challenges=_CH
        )
        verifier = SpartanVerifier(
            _VerifierOnlyPcs(pcs),
            outer=outer_v,
            inner=inner_v,
            witness_open=opening_v,
            challenges=_CH,
        )
        claim = SpartanClaim(instance, io)
        proved = prover.prove(claim, SpartanWitness(z), cheap_transcript(KB))
        verified = verifier.verify(claim, proved.reduction_proof, cheap_transcript(KB))
        self.assertTrue(bool(verified.ok))
        for prover_boundary, verifier_boundary in (
            (outer_p, outer_v),
            (inner_p, inner_v),
            (opening_p, opening_v),
        ):
            _assert_boundary_agrees(self, prover_boundary, verifier_boundary)
        _, prover_next = proved.transcript.sample(1)
        _, verifier_next = verified.transcript.sample(1)
        self.assertTrue(bool(fnp.all(prover_next == verifier_next)))

    def test_various_shapes(self) -> None:
        for seed, (s_x, nvp, num_io) in enumerate(
            [(1, 2, 1), (4, 4, 3), (2, 8, 0), (3, 2, 1)]
        ):
            with self.subTest(s_x=s_x, nvp=nvp, num_io=num_io):
                _, _, _, _, ok = _prove_verify(100 + seed, s_x, nvp, num_io)
                self.assertTrue(bool(ok))

    def test_tampered_outer_claim_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(5, 3, 4, 2)
        bad = replace(
            proof,
            outer=replace(
                proof.outer,
                claims=proof.outer.claims.at[1].add(fnp.ones((), KB)),
            ),
        )
        self.assertFalse(
            bool(
                SpartanVerifier(pcs, challenges=_CH)
                .verify(SpartanClaim(instance, io), bad, cheap_transcript(KB))
                .ok
            )
        )

    def test_tampered_witness_opening_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(6, 3, 4, 2)
        bad = replace(
            proof,
            witness_open=replace(
                proof.witness_open,
                values=proof.witness_open.values.at[0].add(fnp.ones((), KB)),
            ),
        )
        self.assertFalse(
            bool(
                SpartanVerifier(pcs, challenges=_CH)
                .verify(SpartanClaim(instance, io), bad, cheap_transcript(KB))
                .ok
            )
        )

    def test_wrong_witness_commitment_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(8, 3, 4, 2)
        bad = replace(proof, commitment=proof.commitment.at[0].add(fnp.ones((), KB)))
        self.assertFalse(
            bool(
                SpartanVerifier(pcs, challenges=_CH)
                .verify(SpartanClaim(instance, io), bad, cheap_transcript(KB))
                .ok
            )
        )

    def test_unsatisfying_witness_rejected(self) -> None:
        instance, z, _, io = toy_r1cs(9, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        claim = SpartanClaim(instance, io)
        proof = (
            SpartanProver(pcs, challenges=_CH)
            .prove(
                claim,
                SpartanWitness(z.at[0].add(fnp.ones((), KB))),
                cheap_transcript(KB),
            )
            .reduction_proof
        )
        verified = SpartanVerifier(_VerifierOnlyPcs(pcs), challenges=_CH).verify(
            claim, proof, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_truncated_outer_sumcheck_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(10, 3, 4, 2)
        bad = replace(
            proof, outer=replace(proof.outer, sumcheck=proof.outer.sumcheck[:-1])
        )
        with self.assertRaises(ValueError):
            SpartanVerifier(pcs, challenges=_CH).verify(
                SpartanClaim(instance, io), bad, cheap_transcript(KB)
            )

    def test_truncated_inner_sumcheck_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(11, 3, 4, 2)
        bad = replace(
            proof, inner=replace(proof.inner, sumcheck=proof.inner.sumcheck[:-1])
        )
        with self.assertRaises(ValueError):
            SpartanVerifier(pcs, challenges=_CH).verify(
                SpartanClaim(instance, io), bad, cheap_transcript(KB)
            )

    def test_extension_challenge_policy_roundtrips(self) -> None:
        instance, z, _, io = toy_r1cs(12, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        policy = ChallengePolicy(KBX4)
        claim = SpartanClaim(instance, io)
        proved = SpartanProver(pcs, challenges=policy).prove(
            claim, SpartanWitness(z), cheap_transcript(KB)
        )
        verified = SpartanVerifier(_VerifierOnlyPcs(pcs), challenges=policy).verify(
            claim, proved.reduction_proof, cheap_transcript(KB)
        )
        self.assertTrue(bool(verified.ok))

    def test_claim_framing_binds_r1cs_index(self) -> None:
        instance, _, _, io = toy_r1cs(13, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        commitment = fnp.arange(instance.num_vars_padded, dtype=KB)
        changed = replace(instance, a=instance.a.at[0, 0].add(fnp.ones((), KB)))
        t0 = _absorb_claim(cheap_transcript(KB), SpartanClaim(instance, io), commitment)
        t1 = _absorb_claim(cheap_transcript(KB), SpartanClaim(changed, io), commitment)
        _, c0 = t0.sample(1)
        _, c1 = t1.sample(1)
        self.assertFalse(bool(fnp.all(c0 == c1)))

    def test_frames_distinguish_same_values_under_different_splits(self) -> None:
        values = fnp.array([1, 2, 3], KB)
        left = _observe_framed(cheap_transcript(KB), 1, values[:1])
        left = _observe_framed(left, 2, values[1:])
        right = _observe_framed(cheap_transcript(KB), 1, values[:2])
        right = _observe_framed(right, 2, values[2:])
        _, c0 = left.sample(1)
        _, c1 = right.sample(1)
        self.assertFalse(bool(fnp.all(c0 == c1)))


if __name__ == "__main__":
    absltest.main()
