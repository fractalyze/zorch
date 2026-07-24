# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The paired outer stage over two prime fields."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import Array

from zorch.poly.multilinear import eval_mle
from zorch.spartan.r1cs import R1CS
from zorch.spartan.summand import ZerocheckSummand
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import (
    OuterOutput,
    OuterPolynomials,
    OuterProof,
    OuterStage,
)
from zorch.stage import ProveResult
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont
BB = zk_dtypes.babybear_mont
FIELDS = (("koalabear", KB), ("babybear", BB))


class OuterStageTest(parameterized.TestCase):
    def _prove(
        self, seed: int, dtype: Any
    ) -> tuple[R1CS, Array, ProveResult[OuterOutput, OuterProof]]:
        inst, z, _, _ = toy_r1cs(seed, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        az, bz, cz = inst.matvecs(z)
        result = OuterStage().prove(
            OuterPolynomials(az, bz, cz), cheap_transcript(dtype)
        )
        return inst, z, result

    @parameterized.named_parameters(*FIELDS)
    def test_roundtrip_accepts(self, dtype: Any) -> None:
        _, _, proved = self._prove(1, dtype)
        verified = OuterStage().verify(None, proved.proof, cheap_transcript(dtype))
        self.assertTrue(bool(verified.ok))
        self.assertTrue(bool(fnp.all(proved.output.point == verified.output.point)))
        self.assertTrue(bool(fnp.all(proved.output.claims == verified.output.claims)))

    @parameterized.named_parameters(*FIELDS)
    def test_claims_are_matvec_evals(self, dtype: Any) -> None:
        inst, z, proved = self._prove(7, dtype)
        az, bz, cz = inst.matvecs(z)
        want = fnp.stack(
            [
                eval_mle(az, proved.output.point),
                eval_mle(bz, proved.output.point),
                eval_mle(cz, proved.output.point),
            ]
        )
        self.assertTrue(bool(fnp.all(proved.output.claims == want)))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_claim_rejected(self, dtype: Any) -> None:
        _, _, proved = self._prove(2, dtype)
        bad = replace(
            proved.proof,
            claims=proved.proof.claims.at[0].add(fnp.ones((), dtype)),
        )
        verified = OuterStage().verify(None, bad, cheap_transcript(dtype))
        self.assertFalse(bool(verified.ok))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_round_poly_rejected(self, dtype: Any) -> None:
        _, _, proved = self._prove(3, dtype)
        bad = replace(
            proved.proof,
            round_polys=proved.proof.round_polys.at[0, 0].add(fnp.ones((), dtype)),
        )
        verified = OuterStage().verify(None, bad, cheap_transcript(dtype))
        self.assertFalse(bool(verified.ok))

    def test_summand_round_poly_is_fusion_ready(self) -> None:
        from zorch.sumcheck.prover import StandardRound

        stacked = fnp.stack([rand_field(60 + i, (8,), KB) for i in range(4)])
        assert_fusion_ready(
            StandardRound(ZerocheckSummand())._round_poly, stacked, reduces=1
        )


if __name__ == "__main__":
    absltest.main()
