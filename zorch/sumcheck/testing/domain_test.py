# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.sumcheck.domain import (
    EvalDomain,
    extend_to_round_domain,
    fold_stacked,
    product_round_coeffs,
    product_round_poly,
)
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


class DomainTest(absltest.TestCase):
    def test_extend_to_round_domain_nodes(self) -> None:
        # p(0)=2, p(1)=5 → slope 3: U_3 = [∞=3, 0=2, 1=5, 2=8]; Û_3 drops p(1).
        p0, p1 = jnp.array([2], dtype=KB), jnp.array([5], dtype=KB)
        full = extend_to_round_domain(p0, p1, 3)
        self.assertTrue(bool(jnp.array_equal(full[:, 0], jnp.array([3, 2, 5, 8], KB))))
        uhat = extend_to_round_domain(p0, p1, 3, skip_one=True)
        self.assertTrue(bool(jnp.array_equal(uhat[:, 0], jnp.array([3, 2, 8], KB))))

    def test_product_round_poly_two_factors(self) -> None:
        # s(u) = Σ_x' (a0+u·da)(b0+u·db) over Û_2 = [s(∞), s(0)].
        a = jnp.array([1, 2, 3, 4], dtype=KB)
        b = jnp.array([5, 6, 7, 8], dtype=KB)
        msg = product_round_poly(jnp.stack([a, b]))
        a0, da = a[:2], a[2:] - a[:2]
        b0, db = b[:2], b[2:] - b[:2]
        self.assertTrue(bool(msg[0] == jnp.sum(da * db)))  # u = ∞
        self.assertTrue(bool(msg[1] == jnp.sum(a0 * b0)))  # u = 0

    def test_round_domain_to_coeffs(self) -> None:
        # p(x) = 1 + 2x + 3x², so p(0),p(1),p(2) = 1,6,17 and leading coeff = 3.
        want = jnp.array([1, 2, 3], dtype=KB)
        # Finite domain {0,1,2}: recover coeffs from the three evaluations.
        finite = EvalDomain(jnp.array([0, 1, 2], dtype=KB))
        self.assertTrue(
            bool(
                jnp.array_equal(finite.to_coeffs(jnp.array([1, 6, 17], dtype=KB)), want)
            )
        )
        # ∞-led domain [∞,0,1]: recover from [leading, p(0), p(1)] = [3, 1, 6]; the
        # degree and field come from the values, so the domain takes no args.
        led = EvalDomain(leading=True)
        self.assertTrue(
            bool(jnp.array_equal(led.to_coeffs(jnp.array([3, 1, 6], dtype=KB)), want))
        )

    def test_sp1_node_sets_to_coeffs(self) -> None:
        # EvalDomain is node-set-generic, so the SP1 engines slated for zorch reuse
        # it directly: materialized logup samples {0, 1/2} (degree 1), zerocheck
        # samples {0, 2, 4} (degree 2). Both recover coefficients here.
        half = jnp.array(1, dtype=KB) / jnp.array(2, dtype=KB)
        # p(x) = 3 + 5x on {0, 1/2}: p(0)=3, p(1/2)=3 + 5/2.
        logup = EvalDomain(jnp.stack([jnp.array(0, dtype=KB), half]))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    logup.to_coeffs(jnp.stack([jnp.array(3, dtype=KB), 3 + 5 * half])),
                    jnp.array([3, 5], dtype=KB),
                )
            )
        )
        # p(x) = 1 + 2x + 3x² on {0, 2, 4}: p = 1, 17, 57.
        zerocheck = EvalDomain(jnp.array([0, 2, 4], dtype=KB))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    zerocheck.to_coeffs(jnp.array([1, 17, 57], dtype=KB)),
                    jnp.array([1, 2, 3], dtype=KB),
                )
            )
        )

    def test_product_coeffs_verify(self) -> None:
        # The coefficient form of each product round verifies against
        # CoeffsSumcheckRound and reduces the claim down to Π_k P_k(r).
        m, l = 2, 4
        P = jnp.arange(1, m * (1 << l) + 1, dtype=KB).reshape(m, 1 << l)
        claim = jnp.sum(jnp.prod(P, axis=0))
        state = P
        verifier = CoeffsSumcheckRound(degree=m)
        transcript: Transcript = cheap_transcript(KB)
        point = []
        for _ in range(l):
            coeffs = product_round_coeffs(state)
            self.assertEqual(coeffs.shape, (m + 1,))
            claim, transcript, r, ok = verifier(claim, coeffs, transcript)
            self.assertTrue(bool(ok))
            state = fold_stacked(state, r)
            point.append(r)
        self.assertTrue(bool(claim == jnp.prod(state[:, 0])))

    def test_product_coeffs_single_factor(self) -> None:
        # m=1 is degree 1 (U_1 = [∞, 0]) → 2 coeffs [s(0), slope]; p0=1..4, p1=5..8
        # give s(0)=Σp0=10, s(1)=Σp1=26, slope=16.
        p = jnp.arange(1, 9, dtype=KB).reshape(1, 8)
        got = product_round_coeffs(p)
        self.assertTrue(bool(jnp.array_equal(got, jnp.array([10, 16], dtype=KB))))


if __name__ == "__main__":
    absltest.main()
