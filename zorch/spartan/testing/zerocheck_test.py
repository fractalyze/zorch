# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The paired outer stage over two prime fields."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.poly.multilinear import eval_mle
from zorch.spartan.r1cs import R1CS
from zorch.spartan.summand import ZerocheckSummand
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import (
    OuterProof,
    OuterProver,
    OuterVerifier,
    RowEvaluationClaim,
    ZerocheckClaim,
    ZerocheckWitness,
)
from zorch.stage import ProveResult
from zorch.sumcheck.domain import natural_domain
from zorch.sumcheck.eq.eq_poly import EqPolyRound
from zorch.sumcheck.eq.stage import EqPolyProver, EqPolyWitness, EqSumClaim
from zorch.sumcheck.stage import EvaluationClaim
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont
KBX4 = zk_dtypes.koalabearx4_mont
BB = zk_dtypes.babybear_mont
FIELDS = (("koalabear", KB), ("babybear", BB))


class _RecordingEqPolyProver(EqPolyProver):
    def __init__(self) -> None:
        super().__init__(ZerocheckSummand(), challenges=ChallengePolicy(KBX4))
        self.claim: EqSumClaim | None = None
        self.witness: EqPolyWitness | None = None

    def prove(
        self,
        claim: EqSumClaim,
        witness: EqPolyWitness,
        transcript: Transcript,
    ) -> ProveResult[EvaluationClaim, Array]:
        self.claim = claim
        self.witness = witness
        return super().prove(claim, witness, transcript)


class OuterRoleTest(parameterized.TestCase):
    def _prove(
        self, seed: int, dtype: Any
    ) -> tuple[R1CS, Array, ProveResult[RowEvaluationClaim, OuterProof]]:
        inst, z, _, _ = toy_r1cs(seed, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        az, bz, cz = inst.matvecs(z)
        result = OuterProver(challenges=ChallengePolicy(dtype)).prove(
            ZerocheckClaim(inst.s_x),
            ZerocheckWitness(az, bz, cz),
            cheap_transcript(dtype),
        )
        return inst, z, result

    @parameterized.named_parameters(*FIELDS)
    def test_roundtrip_accepts(self, dtype: Any) -> None:
        inst, _, proved = self._prove(1, dtype)
        verified = OuterVerifier(challenges=ChallengePolicy(dtype)).verify(
            ZerocheckClaim(inst.s_x), proved.reduction_proof, cheap_transcript(dtype)
        )
        self.assertTrue(bool(verified.ok))
        self.assertTrue(
            bool(fnp.all(proved.reduced_claim.point == verified.reduced_claim.point))
        )
        self.assertTrue(
            bool(fnp.all(proved.reduced_claim.values == verified.reduced_claim.values))
        )

    @parameterized.named_parameters(*FIELDS)
    def test_claims_are_matvec_evals(self, dtype: Any) -> None:
        inst, z, proved = self._prove(7, dtype)
        az, bz, cz = inst.matvecs(z)
        want = fnp.stack(
            [
                eval_mle(az, proved.reduced_claim.point),
                eval_mle(bz, proved.reduced_claim.point),
                eval_mle(cz, proved.reduced_claim.point),
            ]
        )
        self.assertTrue(bool(fnp.all(proved.reduced_claim.values == want)))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_claim_rejected(self, dtype: Any) -> None:
        inst, _, proved = self._prove(2, dtype)
        bad = replace(
            proved.reduction_proof,
            claims=proved.reduction_proof.claims.at[0].add(fnp.ones((), dtype)),
        )
        verified = OuterVerifier(challenges=ChallengePolicy(dtype)).verify(
            ZerocheckClaim(inst.s_x), bad, cheap_transcript(dtype)
        )
        self.assertFalse(bool(verified.ok))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_round_poly_rejected(self, dtype: Any) -> None:
        inst, _, proved = self._prove(3, dtype)
        bad = replace(
            proved.reduction_proof,
            sumcheck=proved.reduction_proof.sumcheck.at[0, 0].add(fnp.ones((), dtype)),
        )
        verified = OuterVerifier(challenges=ChallengePolicy(dtype)).verify(
            ZerocheckClaim(inst.s_x), bad, cheap_transcript(dtype)
        )
        self.assertFalse(bool(verified.ok))

    def test_extension_keeps_round_zero_factors_in_base_field(self) -> None:
        inst, z, _, _ = toy_r1cs(8, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        az, bz, cz = inst.matvecs(z)
        sumcheck = _RecordingEqPolyProver()
        OuterProver(sumcheck=sumcheck, challenges=ChallengePolicy(KBX4)).prove(
            ZerocheckClaim(inst.s_x),
            ZerocheckWitness(az, bz, cz),
            cheap_transcript(KB),
        )
        assert sumcheck.claim is not None
        assert sumcheck.witness is not None
        self.assertEqual(sumcheck.witness.factors.dtype, KB)
        self.assertEqual(sumcheck.claim.equality_point.dtype, KBX4)

    def test_summand_round_poly_is_fusion_ready(self) -> None:
        stacked = fnp.stack([rand_field(60 + i, (8,), KB) for i in range(3)])
        tau = rand_field(64, (3,), KB)
        round_ = EqPolyRound(
            ZerocheckSummand(),
            tau,
            natural_domain(3, KB),
            challenges=ChallengePolicy(KB),
        )
        state = (stacked, fnp.ones(1, KB))
        assert_fusion_ready(
            lambda value: round_._round_poly(value)[0], state, reduces=1
        )


if __name__ == "__main__":
    absltest.main()
