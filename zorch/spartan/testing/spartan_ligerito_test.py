# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end Spartan driven by the real Ligerito recursive PCS (not `DensePcs`),
proving the witness-open glue is genuinely PCS-agnostic across the seam."""

from __future__ import annotations

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.ligerito.config import LigeritoConfig
from zorch.spartan import spartan
from zorch.spartan.testing.ligerito_pcs import (
    LigeritoSpartanProver,
    LigeritoSpartanVerifier,
)
from zorch.spartan.testing.toy import toy_r1cs
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont

# num_vars = log2(num_vars_padded) = s_y - 1 = the witness-open point dimension.
_CFG = LigeritoConfig(num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=(4, 4))


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


class SpartanLigeritoTest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        # num_vars_padded = 16 = 2^_CFG.num_vars so W opens at r_y[1:] (4 vars).
        inst, z, _, io = toy_r1cs(1, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        proof, _ = spartan.prove(
            inst, z, io, LigeritoSpartanProver(_CFG), _transcript()
        )
        ok, _ = spartan.verify(
            inst, io, proof, LigeritoSpartanVerifier(_CFG), _transcript()
        )
        self.assertTrue(bool(ok))

    def test_tampered_witness_opening_rejected(self) -> None:
        inst, z, _, io = toy_r1cs(2, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        proof, _ = spartan.prove(
            inst, z, io, LigeritoSpartanProver(_CFG), _transcript()
        )
        values, pf = proof.messages[3]
        bad_msgs = list(proof.messages)
        bad_msgs[3] = (values.at[0].add(jnp.ones((), KB)), pf)
        bad = spartan.SpartanProof(proof.commitment, bad_msgs)
        ok, _ = spartan.verify(
            inst, io, bad, LigeritoSpartanVerifier(_CFG), _transcript()
        )
        self.assertFalse(bool(ok))

    def test_unsatisfying_witness_rejected(self) -> None:
        inst, z, _, io = toy_r1cs(3, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        bad_z = z.at[0].add(jnp.ones((), KB))
        proof, _ = spartan.prove(
            inst, bad_z, io, LigeritoSpartanProver(_CFG), _transcript()
        )
        ok, _ = spartan.verify(
            inst, io, proof, LigeritoSpartanVerifier(_CFG), _transcript()
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
