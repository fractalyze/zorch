# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.spartan import spartan
from zorch.spartan.pcs_glue import DensePcs
from zorch.spartan.r1cs import R1CS
from zorch.spartan.testing.toy import toy_r1cs
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _prove_verify(
    seed: int, s_x: int, nvp: int, num_io: int
) -> tuple[R1CS, Array, spartan.SpartanProof, DensePcs, Array]:
    inst, z, _, io = toy_r1cs(
        seed, s_x=s_x, num_vars_padded=nvp, num_io=num_io, dtype=KB
    )
    pcs = DensePcs()
    proof, _ = spartan.prove(inst, z, io, pcs, cheap_transcript(KB))
    ok, _ = spartan.verify(inst, io, proof, pcs, cheap_transcript(KB))
    return inst, io, proof, pcs, ok


class SpartanE2ETest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        _, _, _, _, ok = _prove_verify(1, s_x=3, nvp=4, num_io=2)
        self.assertTrue(bool(ok))

    def test_various_shapes(self) -> None:
        for seed, (s_x, nvp, num_io) in enumerate(
            [(2, 2, 1), (4, 8, 3), (1, 2, 0), (3, 4, 2)]
        ):
            with self.subTest(s_x=s_x, nvp=nvp, num_io=num_io):
                _, _, _, _, ok = _prove_verify(100 + seed, s_x, nvp, num_io)
                self.assertTrue(bool(ok))

    def test_tampered_outer_claim_rejected(self) -> None:
        inst, io, proof, pcs, _ = _prove_verify(5, s_x=3, nvp=4, num_io=2)
        round_polys, claims = proof.messages[0]
        bad_msgs = list(proof.messages)
        bad_msgs[0] = (round_polys, claims.at[1].add(jnp.ones((), KB)))
        bad = spartan.SpartanProof(proof.commitment, bad_msgs)
        ok, _ = spartan.verify(inst, io, bad, pcs, cheap_transcript(KB))
        self.assertFalse(bool(ok))

    def test_tampered_witness_opening_rejected(self) -> None:
        inst, io, proof, pcs, _ = _prove_verify(6, s_x=3, nvp=4, num_io=2)
        values, pf = proof.messages[3]
        bad_msgs = list(proof.messages)
        bad_msgs[3] = (values.at[0].add(jnp.ones((), KB)), pf)
        bad = spartan.SpartanProof(proof.commitment, bad_msgs)
        ok, _ = spartan.verify(inst, io, bad, pcs, cheap_transcript(KB))
        self.assertFalse(bool(ok))

    def test_wrong_witness_commitment_rejected(self) -> None:
        # A commitment to a different witness fails the opening recomputation.
        inst, io, proof, pcs, _ = _prove_verify(8, s_x=3, nvp=4, num_io=2)
        bad = spartan.SpartanProof(
            proof.commitment.at[0].add(jnp.ones((), KB)), proof.messages
        )
        ok, _ = spartan.verify(inst, io, bad, pcs, cheap_transcript(KB))
        self.assertFalse(bool(ok))

    def test_unsatisfying_witness_rejected(self) -> None:
        # Perturb one witness entry: the committed W and z no longer satisfy R1CS.
        inst, z, _, io = toy_r1cs(9, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        bad_z = z.at[0].add(jnp.ones((), KB))
        pcs = DensePcs()
        proof, _ = spartan.prove(inst, bad_z, io, pcs, cheap_transcript(KB))
        ok, _ = spartan.verify(inst, io, proof, pcs, cheap_transcript(KB))
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
