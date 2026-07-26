# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The batch operation and paired inner stage over two prime fields."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.spartan.lincheck import (
    BatchedClaims,
    InnerProver,
    InnerVerifier,
    LincheckClaim,
    LincheckWitness,
    _joint_claim,
    batch_claims,
)
from zorch.spartan.r1cs import R1CS
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import RowEvaluationClaim
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont
BB = zk_dtypes.babybear_mont
FIELDS = (("koalabear", KB), ("babybear", BB))


class RlcOperationTest(parameterized.TestCase):
    @parameterized.named_parameters(*FIELDS)
    def test_joint_claim_is_powers_of_r(self, dtype: Any) -> None:
        claims = rand_field(1, (3,), dtype)
        r = fnp.asarray(rand_field(2, (1,), dtype)[0])
        self.assertTrue(
            bool(
                _joint_claim(claims, r) == claims[0] + r * claims[1] + r * r * claims[2]
            )
        )

    @parameterized.named_parameters(*FIELDS)
    def test_same_operation_replays_for_prover_and_verifier(self, dtype: Any) -> None:
        claims = rand_field(3, (3,), dtype)
        prover, _ = batch_claims(
            claims, cheap_transcript(dtype), challenges=ChallengePolicy(dtype)
        )
        verifier, _ = batch_claims(
            claims, cheap_transcript(dtype), challenges=ChallengePolicy(dtype)
        )
        self.assertTrue(bool(prover.challenge == verifier.challenge))
        self.assertTrue(bool(prover.joint == verifier.joint))


class InnerRoleTest(parameterized.TestCase):
    def _setup(
        self, seed: int, dtype: Any
    ) -> tuple[R1CS, Array, RowEvaluationClaim, BatchedClaims]:
        inst, z, _, _ = toy_r1cs(seed, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        point = rand_field(seed + 10, (inst.s_x,), dtype)
        r = fnp.asarray(rand_field(seed + 11, (1,), dtype)[0])
        joint = fnp.sum(inst.combined_row_mle(point, r) * z)
        return (
            inst,
            z,
            RowEvaluationClaim(point, rand_field(seed + 12, (3,), dtype)),
            BatchedClaims(r, joint),
        )

    @parameterized.named_parameters(*FIELDS)
    def test_roundtrip_accepts(self, dtype: Any) -> None:
        inst, z, outer, batch = self._setup(20, dtype)
        claim = LincheckClaim(inst, outer, batch)
        proved = InnerProver(challenges=ChallengePolicy(dtype)).prove(
            claim, LincheckWitness(z), cheap_transcript(dtype)
        )
        verified = InnerVerifier(challenges=ChallengePolicy(dtype)).verify(
            claim, proved.reduction_proof, cheap_transcript(dtype)
        )
        self.assertTrue(bool(verified.ok))
        self.assertTrue(
            bool(fnp.all(proved.reduced_claim.point == verified.reduced_claim.point))
        )
        self.assertTrue(
            bool(proved.reduced_claim.value == verified.reduced_claim.value)
        )

    @parameterized.named_parameters(*FIELDS)
    def test_wrong_joint_claim_rejected(self, dtype: Any) -> None:
        inst, z, outer, batch = self._setup(21, dtype)
        claim = LincheckClaim(inst, outer, batch)
        proved = InnerProver(challenges=ChallengePolicy(dtype)).prove(
            claim, LincheckWitness(z), cheap_transcript(dtype)
        )
        bad_batch = replace(batch, joint=batch.joint + fnp.ones((), dtype))
        bad_claim = replace(claim, batch=bad_batch)
        verified = InnerVerifier(challenges=ChallengePolicy(dtype)).verify(
            bad_claim, proved.reduction_proof, cheap_transcript(dtype)
        )
        self.assertFalse(bool(verified.ok))


if __name__ == "__main__":
    absltest.main()
