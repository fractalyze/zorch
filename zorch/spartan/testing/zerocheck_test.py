# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The outer (zerocheck) stage, over two prime fields.

The stage carries no field of its own — dtype flows from the matvecs — so the
prove/verify roundtrip and its soundness checks run over both koalabear and
babybear, pinning the field-agnosticism against a single-field assumption
creeping in.
"""
from __future__ import annotations

from typing import Any, cast

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
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
BB = zk_dtypes.babybear_mont
FIELDS = (("koalabear", KB), ("babybear", BB))


def _prove_outer(
    inst: R1CS, z: Array, dtype: Any
) -> tuple[SpartanCarry, tuple[Array, Array]]:
    az, bz, cz = inst.matvecs(z)
    carry, _, msg = OuterProver(az, bz, cz)(SpartanCarry(), cheap_transcript(dtype))
    return carry, msg


class OuterStageTest(parameterized.TestCase):
    @parameterized.named_parameters(*FIELDS)
    def test_roundtrip_accepts(self, dtype: Any) -> None:
        inst, z, _, _ = toy_r1cs(1, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        pcarry, msg = _prove_outer(inst, z, dtype)
        vcarry, _, ok = OuterVerifier()(SpartanCarry(), msg, cheap_transcript(dtype))
        self.assertTrue(bool(ok))
        # prover and verifier agree on the bound point and the claimed evals.
        self.assertTrue(bool(jnp.all(pcarry.r_x == vcarry.r_x)))
        self.assertTrue(bool(jnp.all(pcarry.claims_outer == vcarry.claims_outer)))

    @parameterized.named_parameters(*FIELDS)
    def test_claims_are_the_matvec_evals_at_r_x(self, dtype: Any) -> None:
        # claims_outer == (Az, Bz, Cz)(r_x), the outer sumcheck's terminal evals.
        from zorch.poly.multilinear import eval_mle

        inst, z, _, _ = toy_r1cs(7, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        carry, _ = _prove_outer(inst, z, dtype)
        r_x = cast(Array, carry.r_x)
        az, bz, cz = inst.matvecs(z)
        want = jnp.stack([eval_mle(az, r_x), eval_mle(bz, r_x), eval_mle(cz, r_x)])
        self.assertTrue(bool(jnp.all(carry.claims_outer == want)))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_claim_rejected(self, dtype: Any) -> None:
        inst, z, _, _ = toy_r1cs(2, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        _, (round_polys, claims) = _prove_outer(inst, z, dtype)
        bad = claims.at[0].add(jnp.ones((), dtype))
        _, _, ok = OuterVerifier()(
            SpartanCarry(), (round_polys, bad), cheap_transcript(dtype)
        )
        self.assertFalse(bool(ok))

    @parameterized.named_parameters(*FIELDS)
    def test_tampered_round_poly_rejected(self, dtype: Any) -> None:
        inst, z, _, _ = toy_r1cs(3, s_x=3, num_vars_padded=4, num_io=2, dtype=dtype)
        _, (round_polys, claims) = _prove_outer(inst, z, dtype)
        bad = round_polys.at[0, 0].add(jnp.ones((), dtype))
        _, _, ok = OuterVerifier()(
            SpartanCarry(), (bad, claims), cheap_transcript(dtype)
        )
        self.assertFalse(bool(ok))

    def test_summand_round_poly_is_fusion_ready(self) -> None:
        # The zerocheck round body eq·(â◦b̂−ĉ) lowers to element-wise ops + one Σ.
        from zorch.sumcheck.prover import StandardRound

        a = rand_field(60, (8,), KB)
        b = rand_field(61, (8,), KB)
        c = rand_field(62, (8,), KB)
        e = rand_field(63, (8,), KB)
        rnd = StandardRound(ZerocheckSummand())
        assert_fusion_ready(rnd._round_poly, jnp.stack([e, a, b, c]), reduces=1)


if __name__ == "__main__":
    absltest.main()
