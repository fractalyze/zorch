# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.poly.multilinear import eval_mle
from zorch.spartan.lincheck import BatchedClaims, InnerOutput
from zorch.spartan.pcs_glue import (
    DensePcs,
    WitnessOpenData,
    WitnessOpeningClaim,
    WitnessOpeningStatement,
    WitnessOpenProof,
    WitnessOpenStage,
    witness_opening_claim,
)
from zorch.spartan.r1cs import eval_public_half, recombine_z_eval
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import OuterOutput
from zorch.stage import ProveResult
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


class DensePcsTest(absltest.TestCase):
    def test_open_verify_roundtrip(self) -> None:
        poly = rand_field(1, (8,), KB)
        point = rand_field(2, (3,), KB)
        pcs = DensePcs()
        commitment, data = pcs.commit([poly])
        values, proof, _ = pcs.open(data, [point], cheap_transcript(KB))
        self.assertTrue(bool(values[0] == eval_mle(poly, point)))
        ok, _ = pcs.verify(commitment, [point], values, proof, cheap_transcript(KB))
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        poly = rand_field(3, (8,), KB)
        point = rand_field(4, (3,), KB)
        pcs = DensePcs()
        commitment, _ = pcs.commit([poly])
        bad = fnp.stack([eval_mle(poly, point) + fnp.ones((), KB)])
        ok, _ = pcs.verify(commitment, [point], bad, None, cheap_transcript(KB))
        self.assertFalse(bool(ok))


class WitnessOpenStageTest(absltest.TestCase):
    def _run(self, seed: int) -> tuple[
        WitnessOpenStage,
        DensePcs,
        Array,
        WitnessOpeningClaim,
        ProveResult[None, WitnessOpenProof],
    ]:
        instance, z, _, io = toy_r1cs(
            seed, s_x=3, num_vars_padded=4, num_io=2, dtype=KB
        )
        pcs = DensePcs()
        commitment, data = pcs.commit([z[: instance.num_vars_padded]])
        point_x = rand_field(seed + 10, (instance.s_x,), KB)
        challenge = fnp.asarray(rand_field(seed + 11, (1,), KB)[0])
        point_y = rand_field(seed + 12, (instance.s_y,), KB)
        outer = OuterOutput(point_x, rand_field(seed + 13, (3,), KB))
        batch = BatchedClaims(challenge, rand_field(seed + 14, (), KB))
        final = instance.eval_combined_matrix(point_x, point_y, challenge) * eval_mle(
            z, point_y
        )
        inner = InnerOutput(point_y, final)
        stage = WitnessOpenStage()
        claim = witness_opening_claim(instance, io, outer, batch, inner)
        proved = stage.prove(
            WitnessOpenData(pcs, data, inner.point), cheap_transcript(KB)
        )
        return stage, pcs, commitment, claim, proved

    def test_roundtrip_accepts(self) -> None:
        stage, pcs, commitment, claim, proved = self._run(5)
        verified = stage.verify(
            WitnessOpeningStatement(pcs, commitment, claim),
            proved.proof,
            cheap_transcript(KB),
        )
        self.assertTrue(bool(verified.ok))

    def test_wrong_inner_final_rejected(self) -> None:
        stage, pcs, commitment, claim, proved = self._run(6)
        bad = replace(claim, final_claim=claim.final_claim + fnp.ones((), KB))
        verified = stage.verify(
            WitnessOpeningStatement(pcs, commitment, bad),
            proved.proof,
            cheap_transcript(KB),
        )
        self.assertFalse(bool(verified.ok))

    def test_public_half_reconstruction(self) -> None:
        _, z, _, io = toy_r1cs(7, s_x=2, num_vars_padded=4, num_io=2, dtype=KB)
        point = rand_field(30, (3,), KB)
        eval_w = eval_mle(z[:4], point[1:])
        eval_pub = eval_public_half(io, point[1:], 4)
        self.assertTrue(
            bool(recombine_z_eval(eval_w, eval_pub, point[0]) == eval_mle(z, point))
        )


if __name__ == "__main__":
    absltest.main()
