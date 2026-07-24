# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end Spartan driven by the real Ligerito recursive PCS."""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.ligerito.config import LigeritoConfig
from zorch.spartan.spartan import Spartan, SpartanStatement, SpartanWitness
from zorch.spartan.testing.ligerito_pcs import (
    LigeritoSpartanProver,
    LigeritoSpartanVerifier,
)
from zorch.spartan.testing.toy import toy_r1cs
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont
_CFG = LigeritoConfig(num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=(4, 4))


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


class SpartanLigeritoTest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        instance, z, _, io = toy_r1cs(1, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        protocol = Spartan()
        proved = protocol.prove(
            SpartanWitness(instance, z, io, LigeritoSpartanProver(_CFG)),
            _transcript(),
        )
        verified = protocol.verify(
            SpartanStatement(instance, io, LigeritoSpartanVerifier(_CFG)),
            proved.proof,
            _transcript(),
        )
        self.assertTrue(bool(verified.ok))

    def test_tampered_witness_opening_rejected(self) -> None:
        instance, z, _, io = toy_r1cs(2, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        protocol = Spartan()
        proof = protocol.prove(
            SpartanWitness(instance, z, io, LigeritoSpartanProver(_CFG)),
            _transcript(),
        ).proof
        bad = replace(
            proof,
            witness_open=replace(
                proof.witness_open,
                values=proof.witness_open.values.at[0].add(fnp.ones((), KB)),
            ),
        )
        verified = protocol.verify(
            SpartanStatement(instance, io, LigeritoSpartanVerifier(_CFG)),
            bad,
            _transcript(),
        )
        self.assertFalse(bool(verified.ok))

    def test_unsatisfying_witness_rejected(self) -> None:
        instance, z, _, io = toy_r1cs(3, s_x=3, num_vars_padded=16, num_io=2, dtype=KB)
        protocol = Spartan()
        proof = protocol.prove(
            SpartanWitness(
                instance,
                z.at[0].add(fnp.ones((), KB)),
                io,
                LigeritoSpartanProver(_CFG),
            ),
            _transcript(),
        ).proof
        verified = protocol.verify(
            SpartanStatement(instance, io, LigeritoSpartanVerifier(_CFG)),
            proof,
            _transcript(),
        )
        self.assertFalse(bool(verified.ok))


if __name__ == "__main__":
    absltest.main()
