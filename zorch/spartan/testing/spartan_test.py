# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.spartan.lincheck import InnerStage
from zorch.spartan.pcs_glue import WitnessOpenStage
from zorch.spartan.r1cs import R1CS
from zorch.spartan.spartan import (
    Spartan,
    SpartanClaim,
    SpartanProof,
    SpartanWitness,
)
from zorch.spartan.testing.dense_pcs import DensePcs
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import OuterStage
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont
KBX4 = zk_dtypes.koalabearx4_mont


def _prove_verify(
    seed: int, s_x: int, nvp: int, num_io: int
) -> tuple[R1CS, Array, SpartanProof, DensePcs, Array]:
    instance, z, _, io = toy_r1cs(
        seed, s_x=s_x, num_vars_padded=nvp, num_io=num_io, dtype=KB
    )
    pcs = DensePcs()
    claim = SpartanClaim(instance, io)
    protocol = Spartan(pcs, pcs)
    proved = protocol.prove(claim, SpartanWitness(z), cheap_transcript(KB))
    verified = protocol.verify(claim, proved.reduction_proof, cheap_transcript(KB))
    return instance, io, proved.reduction_proof, pcs, verified.ok


class _RecordingStage:
    def __init__(self, stage: Any) -> None:
        self.stage = stage
        self.prover_transcript: Transcript | None = None
        self.verifier_transcript: Transcript | None = None
        self.prover_claim: Any = None
        self.verifier_claim: Any = None

    def prove(self, *args: Any) -> Any:
        result = self.stage.prove(*args)
        self.prover_transcript = result.transcript
        self.prover_claim = result.reduced_claim
        return result

    def verify(self, *args: Any) -> Any:
        result = self.stage.verify(*args)
        self.verifier_transcript = result.transcript
        self.verifier_claim = result.reduced_claim
        return result


class SpartanE2ETest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        _, _, _, _, ok = _prove_verify(1, 2, 4, 2)
        self.assertTrue(bool(ok))

    def test_composite_transcripts_agree(self) -> None:
        instance, z, _, io = toy_r1cs(2, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        outer = _RecordingStage(OuterStage())
        inner = _RecordingStage(InnerStage())
        opening = _RecordingStage(WitnessOpenStage(pcs, pcs))
        protocol = Spartan(
            pcs,
            pcs,
            outer=cast(OuterStage, outer),
            inner=cast(InnerStage, inner),
            witness_open=cast(WitnessOpenStage, opening),
        )
        claim = SpartanClaim(instance, io)
        proved = protocol.prove(claim, SpartanWitness(z), cheap_transcript(KB))
        verified = protocol.verify(
            claim,
            proved.reduction_proof,
            cheap_transcript(KB),
        )
        self.assertTrue(bool(verified.ok))
        for boundary in (outer, inner, opening):
            assert boundary.prover_transcript is not None
            assert boundary.verifier_transcript is not None
            _, prover_next = boundary.prover_transcript.sample(1)
            _, verifier_next = boundary.verifier_transcript.sample(1)
            self.assertTrue(bool(fnp.all(prover_next == verifier_next)))
            if boundary.prover_claim is not None:
                for field in boundary.prover_claim.__dataclass_fields__:
                    prover_value = getattr(boundary.prover_claim, field)
                    verifier_value = getattr(boundary.verifier_claim, field)
                    self.assertTrue(bool(fnp.all(prover_value == verifier_value)))
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
        verified = Spartan(pcs, pcs).verify(
            SpartanClaim(instance, io), bad, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_tampered_witness_opening_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(6, 3, 4, 2)
        bad = replace(
            proof,
            witness_open=replace(
                proof.witness_open,
                values=proof.witness_open.values.at[0].add(fnp.ones((), KB)),
            ),
        )
        verified = Spartan(pcs, pcs).verify(
            SpartanClaim(instance, io), bad, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_wrong_witness_commitment_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(8, 3, 4, 2)
        bad = replace(
            proof,
            commitment=proof.commitment.at[0].add(fnp.ones((), KB)),
        )
        verified = Spartan(pcs, pcs).verify(
            SpartanClaim(instance, io), bad, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_unsatisfying_witness_rejected(self) -> None:
        instance, z, _, io = toy_r1cs(9, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        bad_z = z.at[0].add(fnp.ones((), KB))
        pcs = DensePcs()
        claim = SpartanClaim(instance, io)
        protocol = Spartan(pcs, pcs)
        proved = protocol.prove(claim, SpartanWitness(bad_z), cheap_transcript(KB))
        verified = protocol.verify(claim, proved.reduction_proof, cheap_transcript(KB))
        self.assertFalse(bool(verified.ok))

    def test_truncated_outer_sumcheck_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(10, 3, 4, 2)
        bad = replace(
            proof,
            outer=replace(proof.outer, sumcheck=proof.outer.sumcheck[:-1]),
        )
        with self.assertRaises(ValueError):
            Spartan(pcs, pcs).verify(
                SpartanClaim(instance, io), bad, cheap_transcript(KB)
            )

    def test_truncated_inner_sumcheck_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(11, 3, 4, 2)
        bad = replace(
            proof,
            inner=replace(proof.inner, sumcheck=proof.inner.sumcheck[:-1]),
        )
        with self.assertRaises(ValueError):
            Spartan(pcs, pcs).verify(
                SpartanClaim(instance, io), bad, cheap_transcript(KB)
            )

    def test_extension_challenge_policy_roundtrips(self) -> None:
        instance, z, _, io = toy_r1cs(12, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        claim = SpartanClaim(instance, io)
        protocol = Spartan(pcs, pcs, challenges=ChallengePolicy(KBX4))
        proved = protocol.prove(claim, SpartanWitness(z), cheap_transcript(KB))
        verified = protocol.verify(claim, proved.reduction_proof, cheap_transcript(KB))
        self.assertTrue(bool(verified.ok))

    def test_claim_framing_binds_r1cs_index(self) -> None:
        instance, _, _, io = toy_r1cs(13, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        commitment = fnp.arange(instance.num_vars_padded, dtype=KB)
        changed = replace(instance, a=instance.a.at[0, 0].add(fnp.ones((), KB)))
        t0 = Spartan._absorb_claim(
            cheap_transcript(KB), SpartanClaim(instance, io), commitment
        )
        t1 = Spartan._absorb_claim(
            cheap_transcript(KB), SpartanClaim(changed, io), commitment
        )
        _, c0 = t0.sample(1)
        _, c1 = t1.sample(1)
        self.assertFalse(bool(fnp.all(c0 == c1)))

    def test_frames_distinguish_same_values_under_different_splits(self) -> None:
        values = fnp.array([1, 2, 3], KB)
        left = Spartan._observe_framed(cheap_transcript(KB), 1, values[:1])
        left = Spartan._observe_framed(left, 2, values[1:])
        right = Spartan._observe_framed(cheap_transcript(KB), 1, values[:2])
        right = Spartan._observe_framed(right, 2, values[2:])
        _, c0 = left.sample(1)
        _, c1 = right.sample(1)
        self.assertFalse(bool(fnp.all(c0 == c1)))


if __name__ == "__main__":
    absltest.main()
