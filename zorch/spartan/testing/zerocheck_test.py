# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import cast

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.spartan.carry import SpartanCarry
from zorch.spartan.r1cs import R1CS
from zorch.spartan.summand import ZerocheckSummand
from zorch.spartan.testing.toy import toy_r1cs
from zorch.spartan.zerocheck import OuterProver, OuterVerifier
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _prove_outer(inst: R1CS, z: Array) -> tuple[SpartanCarry, tuple[Array, Array]]:
    az, bz, cz = inst.matvecs(z)
    carry, _, msg = OuterProver(az, bz, cz)(SpartanCarry(), cheap_transcript(KB))
    return carry, msg


class OuterStageTest(absltest.TestCase):
    def test_roundtrip_accepts(self) -> None:
        inst, z, _, _ = toy_r1cs(1, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        pcarry, msg = _prove_outer(inst, z)
        vcarry, _, ok = OuterVerifier()(SpartanCarry(), msg, cheap_transcript(KB))
        self.assertTrue(bool(ok))
        # prover and verifier agree on the bound point and the claimed evals.
        self.assertTrue(bool(jnp.all(pcarry.r_x == vcarry.r_x)))
        self.assertTrue(bool(jnp.all(pcarry.claims_outer == vcarry.claims_outer)))

    def test_claims_are_the_matvec_evals_at_r_x(self) -> None:
        # claims_outer == (Az, Bz, Cz)(r_x), the outer sumcheck's terminal evals.
        from zorch.poly.multilinear import eval_mle

        inst, z, _, _ = toy_r1cs(7, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        carry, _ = _prove_outer(inst, z)
        r_x = cast(Array, carry.r_x)
        az, bz, cz = inst.matvecs(z)
        want = jnp.stack([eval_mle(az, r_x), eval_mle(bz, r_x), eval_mle(cz, r_x)])
        self.assertTrue(bool(jnp.all(carry.claims_outer == want)))

    def test_tampered_claim_rejected(self) -> None:
        inst, z, _, _ = toy_r1cs(2, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        _, (round_polys, claims) = _prove_outer(inst, z)
        bad = claims.at[0].add(jnp.ones((), KB))
        _, _, ok = OuterVerifier()(
            SpartanCarry(), (round_polys, bad), cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))

    def test_tampered_round_poly_rejected(self) -> None:
        inst, z, _, _ = toy_r1cs(3, s_x=3, num_vars_padded=4, num_io=2, dtype=KB)
        _, (round_polys, claims) = _prove_outer(inst, z)
        bad = round_polys.at[0, 0].add(jnp.ones((), KB))
        _, _, ok = OuterVerifier()(SpartanCarry(), (bad, claims), cheap_transcript(KB))
        self.assertFalse(bool(ok))

    def test_summand_round_poly_is_fusion_ready(self) -> None:
        # The zerocheck round body E·(A·B−C) lowers to element-wise ops + one Σ.
        from zorch.sumcheck.prover import StandardRound

        a = rand_field(60, (8,), KB)
        b = rand_field(61, (8,), KB)
        c = rand_field(62, (8,), KB)
        e = rand_field(63, (8,), KB)
        rnd = StandardRound(ZerocheckSummand())
        assert_fusion_ready(rnd._round_poly, jnp.stack([e, a, b, c]), reduces=1)


if __name__ == "__main__":
    absltest.main()
