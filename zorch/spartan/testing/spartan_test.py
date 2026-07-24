# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.spartan.pcs_glue import DensePcs
from zorch.spartan.r1cs import R1CS
from zorch.spartan.spartan import (
    Spartan,
    SpartanProof,
    SpartanStatement,
    SpartanWitness,
)
from zorch.spartan.testing.toy import toy_r1cs
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _prove_verify(
    seed: int, s_x: int, nvp: int, num_io: int
) -> tuple[R1CS, Array, SpartanProof, DensePcs, Array]:
    instance, z, _, io = toy_r1cs(
        seed, s_x=s_x, num_vars_padded=nvp, num_io=num_io, dtype=KB
    )
    pcs = DensePcs()
    protocol = Spartan()
    proved = protocol.prove(SpartanWitness(instance, z, io, pcs), cheap_transcript(KB))
    verified = protocol.verify(
        SpartanStatement(instance, io, pcs), proved.proof, cheap_transcript(KB)
    )
    return instance, io, proved.proof, pcs, verified.ok


class SpartanE2ETest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        _, _, _, _, ok = _prove_verify(1, 3, 4, 2)
        self.assertTrue(bool(ok))

    def test_various_shapes(self) -> None:
        for seed, (s_x, nvp, num_io) in enumerate(
            [(2, 2, 1), (4, 8, 3), (1, 2, 0), (3, 4, 2)]
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
        verified = Spartan().verify(
            SpartanStatement(instance, io, pcs), bad, cheap_transcript(KB)
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
        verified = Spartan().verify(
            SpartanStatement(instance, io, pcs), bad, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_wrong_witness_commitment_rejected(self) -> None:
        instance, io, proof, pcs, _ = _prove_verify(8, 3, 4, 2)
        bad = replace(
            proof,
            commitment=proof.commitment.at[0].add(fnp.ones((), KB)),
        )
        verified = Spartan().verify(
            SpartanStatement(instance, io, pcs), bad, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))

    def test_unsatisfying_witness_rejected(self) -> None:
        instance, z, _, io = toy_r1cs(9, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        bad_z = z.at[0].add(fnp.ones((), KB))
        pcs = DensePcs()
        protocol = Spartan()
        proved = protocol.prove(
            SpartanWitness(instance, bad_z, io, pcs), cheap_transcript(KB)
        )
        verified = protocol.verify(
            SpartanStatement(instance, io, pcs), proved.proof, cheap_transcript(KB)
        )
        self.assertFalse(bool(verified.ok))


if __name__ == "__main__":
    absltest.main()
