# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Structural cross-check of the production prover against the brute-force
reference: degrees, per-round eval tuples, running claims, powers-of-`r`
batching, and the final identities."""

from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.spartan import spartan
from zorch.spartan.pcs_glue import DensePcs
from zorch.spartan.r1cs import R1CS
from zorch.spartan.summand import ZerocheckSummand
from zorch.spartan.testing.reference import naive_round_polys, replay_challenges
from zorch.spartan.testing.toy import toy_r1cs
from zorch.sumcheck.prover import ProductSummand, StandardRound
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _prove(
    seed: int, s_x: int, nvp: int, num_io: int
) -> tuple[R1CS, Array, Array, spartan.SpartanProof, dict[str, Array]]:
    inst, z, _, io = toy_r1cs(
        seed, s_x=s_x, num_vars_padded=nvp, num_io=num_io, dtype=KB
    )
    proof, _ = spartan.prove(inst, z, io, DensePcs(), cheap_transcript(KB))
    ch = replay_challenges(
        cheap_transcript(KB),
        proof.commitment,
        io,
        proof.messages[0][0],
        proof.messages[0][1],
        proof.messages[2],
        inst.s_x,
    )
    return inst, z, io, proof, ch


class StructuralCrossCheckTest(absltest.TestCase):
    def test_round_poly_degrees(self) -> None:
        # Hard invariant: outer degree 3 (4 evals), inner degree 2 (3 evals).
        inst, _, _, proof, _ = _prove(1, s_x=3, nvp=4, num_io=2)
        outer_polys = proof.messages[0][0]
        inner_polys = proof.messages[2]
        self.assertEqual(outer_polys.shape, (inst.s_x, 4))
        self.assertEqual(inner_polys.shape, (inst.s_y, 3))

    def test_outer_round_polys_match_reference(self) -> None:
        inst, z, _, proof, ch = _prove(2, s_x=3, nvp=4, num_io=2)
        az, bz, cz = inst.matvecs(z)
        e = expand_eq_to_hypercube(ch["tau"], jnp.ones((), KB))
        ref_polys, final = naive_round_polys(
            [e, az, bz, cz], lambda e, a, b, c: e * (a * b - c), 3, list(ch["r_x"])
        )
        got = proof.messages[0][0]
        for j, (g, ref) in enumerate(zip(got, ref_polys, strict=True)):
            self.assertTrue(bool(jnp.all(g == ref)), f"outer round {j}")
        # final claimed evals == (Az,Bz,Cz)(r_x).
        claims = proof.messages[0][1]
        self.assertTrue(
            bool(jnp.all(claims == jnp.stack([final[1][0], final[2][0], final[3][0]])))
        )

    def test_outer_running_claim_chain(self) -> None:
        # s_j(0)+s_j(1) == claim_j, claim_0 = 0, claim_{j+1} = s_j(r_j).
        from zorch.poly.univariate import eval_univariate

        inst, _, _, proof, ch = _prove(3, s_x=3, nvp=4, num_io=2)
        polys = proof.messages[0][0]
        claim = jnp.zeros((), KB)
        for j in range(inst.s_x):
            s = polys[j]
            self.assertTrue(bool(claim == s[0] + s[1]), f"round {j} sum-check identity")
            claim = eval_univariate(s, ch["r_x"][j])

    def test_batching_is_powers_of_r(self) -> None:
        # joint_claim == vA + r·vB + r²·vC (powers of one challenge).
        inst, z, _, proof, ch = _prove(4, s_x=3, nvp=4, num_io=2)
        va, vb, vc = proof.messages[0][1]
        r = ch["r_batch"]
        want = va + r * vb + r * r * vc
        # recompute joint via the reference: the inner sumcheck's claim_0.
        m = inst.combined_row_mle(ch["r_x"], r)
        got = jnp.sum(m * z)  # Σ_y M(y)·z(y) == joint_claim
        self.assertTrue(bool(got == want))

    def test_inner_round_polys_match_reference(self) -> None:
        inst, z, _, proof, ch = _prove(5, s_x=3, nvp=4, num_io=2)
        m = inst.combined_row_mle(ch["r_x"], ch["r_batch"])
        ref_polys, _ = naive_round_polys([m, z], lambda a, b: a * b, 2, list(ch["r_y"]))
        got = proof.messages[2]
        for j, (g, ref) in enumerate(zip(got, ref_polys, strict=True)):
            self.assertTrue(bool(jnp.all(g == ref)), f"inner round {j}")

    def test_inner_final_identity(self) -> None:
        # inner_final == eval_ABC · z̃(r_y).
        from zorch.poly.univariate import eval_univariate

        inst, z, _, proof, ch = _prove(6, s_x=3, nvp=4, num_io=2)
        # inner_final = last inner round poly at r_y[-1].
        inner_polys = proof.messages[2]
        claim = jnp.sum(inst.combined_row_mle(ch["r_x"], ch["r_batch"]) * z)
        for j in range(inst.s_y):
            claim = eval_univariate(inner_polys[j], ch["r_y"][j])
        eval_abc = inst.eval_combined_matrix(ch["r_x"], ch["r_y"], ch["r_batch"])
        z_eval = eval_mle(z, ch["r_y"])
        self.assertTrue(bool(claim == eval_abc * z_eval))


class StageFusionTest(absltest.TestCase):
    def test_outer_round_body_fuses(self) -> None:
        stacked = jnp.stack([rand_field(70 + i, (8,), KB) for i in range(4)])
        assert_fusion_ready(
            StandardRound(ZerocheckSummand())._round_poly, stacked, reduces=1
        )

    def test_inner_round_body_fuses(self) -> None:
        stacked = jnp.stack([rand_field(80 + i, (8,), KB) for i in range(2)])
        assert_fusion_ready(
            StandardRound(ProductSummand(2))._round_poly, stacked, reduces=1
        )


if __name__ == "__main__":
    absltest.main()
