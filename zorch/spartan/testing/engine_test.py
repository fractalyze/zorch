# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Swapping the injected per-variable sumcheck engine on a Spartan stage."""

from __future__ import annotations

import zk_dtypes
from absl.testing import absltest

from zorch.spartan import spartan
from zorch.spartan.engine import StageSumcheck
from zorch.spartan.pcs_glue import DensePcs
from zorch.spartan.testing.toy import toy_r1cs
from zorch.sumcheck.prover import CompressedProductRound
from zorch.sumcheck.verifier import CompressedCoeffsSumcheckRound
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _compressed_inner() -> StageSumcheck:
    """A non-default inner engine: the compressed `[c_0, c_2]` coefficient wire
    (the linear coeff reconstructed from the running claim)."""
    return StageSumcheck(CompressedProductRound(), CompressedCoeffsSumcheckRound())


class CustomEngineTest(absltest.TestCase):
    def test_compressed_inner_roundtrips(self) -> None:
        inst, z, _, io = toy_r1cs(1, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        eng = _compressed_inner()
        proof, _ = spartan.prove(
            inst, z, io, DensePcs(), cheap_transcript(KB), inner=eng
        )
        # The compressed wire sends 2 coefficients per round, not the 3 evals the
        # default value-form round sends.
        self.assertEqual(proof.messages[2].shape, (inst.s_y, 2))
        ok, _ = spartan.verify(
            inst, io, proof, DensePcs(), cheap_transcript(KB), inner=eng
        )
        self.assertTrue(bool(ok))

    def test_default_inner_wire_is_three_evals(self) -> None:
        inst, z, _, io = toy_r1cs(2, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        proof, _ = spartan.prove(inst, z, io, DensePcs(), cheap_transcript(KB))
        self.assertEqual(proof.messages[2].shape, (inst.s_y, 3))

    def test_engine_mismatch_fails_loud(self) -> None:
        # A compressed-wire proof verified with the default (value-form) engine
        # is a shape mismatch — the pairing is enforced, not silently accepted.
        inst, z, _, io = toy_r1cs(3, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        proof, _ = spartan.prove(
            inst, z, io, DensePcs(), cheap_transcript(KB), inner=_compressed_inner()
        )
        with self.assertRaises(ValueError):
            spartan.verify(inst, io, proof, DensePcs(), cheap_transcript(KB))


if __name__ == "__main__":
    absltest.main()
