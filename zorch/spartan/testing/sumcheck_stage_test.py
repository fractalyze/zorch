# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Swapping an ordinary sumcheck child stage inside a Spartan stage."""

from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.spartan.lincheck import InnerStage
from zorch.spartan.spartan import Spartan, SpartanClaim, SpartanWitness
from zorch.spartan.testing.dense_pcs import DensePcs
from zorch.spartan.testing.toy import toy_r1cs
from zorch.sumcheck.prover import (
    CompressedProductRound,
    ProductSummand,
    StandardRound,
)
from zorch.sumcheck.stage import SumcheckStage, SumcheckWitness, SumClaim
from zorch.sumcheck.verifier import CompressedCoeffsSumcheckRound, SumcheckRound
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _compressed_inner() -> SumcheckStage:
    """A non-default child stage: the compressed `[c_0, c_2]` coefficient wire
    (the linear coeff reconstructed from the running claim)."""
    return SumcheckStage(CompressedProductRound(), CompressedCoeffsSumcheckRound())


class ChildStageTest(absltest.TestCase):
    def test_prove_returns_the_verifier_replayed_transcript(self) -> None:
        state = rand_field(20, (2, 8), KB)
        claim = fnp.sum(state[0] * state[1])
        stage = SumcheckStage(
            StandardRound(ProductSummand(2)),
            SumcheckRound(2),
        )
        source_claim = SumClaim(claim, 3)
        proved = stage.prove(source_claim, SumcheckWitness(state), cheap_transcript(KB))
        verified = stage.verify(
            source_claim, proved.reduction_proof, cheap_transcript(KB)
        )
        _, prover_next = proved.transcript.sample(1)
        _, verifier_next = verified.transcript.sample(1)
        self.assertTrue(bool(fnp.all(prover_next == verifier_next)))

    def test_compressed_inner_roundtrips(self) -> None:
        inst, z, _, io = toy_r1cs(1, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        sumcheck = _compressed_inner()
        pcs = DensePcs()
        protocol = Spartan(pcs, pcs, inner=InnerStage(sumcheck=sumcheck))
        claim = SpartanClaim(inst, io)
        proved = protocol.prove(claim, SpartanWitness(z), cheap_transcript(KB))
        proof = proved.reduction_proof
        # The compressed wire sends 2 coefficients per round, not the 3 evals the
        # default value-form round sends.
        self.assertEqual(proof.inner.sumcheck.shape, (inst.s_y, 2))
        verified = protocol.verify(claim, proof, cheap_transcript(KB))
        self.assertTrue(bool(verified.ok))

    def test_default_inner_wire_is_three_evals(self) -> None:
        inst, z, _, io = toy_r1cs(2, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        proved = Spartan(pcs, pcs).prove(
            SpartanClaim(inst, io), SpartanWitness(z), cheap_transcript(KB)
        )
        proof = proved.reduction_proof
        self.assertEqual(proof.inner.sumcheck.shape, (inst.s_y, 3))

    def test_child_stage_mismatch_fails_loud(self) -> None:
        # A compressed-wire proof verified with the default value-form child stage
        # is a shape mismatch — the pairing is enforced, not silently accepted.
        inst, z, _, io = toy_r1cs(3, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        pcs = DensePcs()
        proof = (
            Spartan(pcs, pcs, inner=InnerStage(sumcheck=_compressed_inner()))
            .prove(SpartanClaim(inst, io), SpartanWitness(z), cheap_transcript(KB))
            .reduction_proof
        )
        with self.assertRaises(ValueError):
            Spartan(pcs, pcs).verify(
                SpartanClaim(inst, io), proof, cheap_transcript(KB)
            )


if __name__ == "__main__":
    absltest.main()
