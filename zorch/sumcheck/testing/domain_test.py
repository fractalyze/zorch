# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.sumcheck.domain import (
    EvalDomain,
    compressed_domain,
    extend_to_round_domain,
    fold,
    product_round_coeffs,
    product_round_poly,
    summand_evals,
    uhat_domain,
)
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.testkit.fusion import assert_fusion_ready
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
        led = EvalDomain(inf_index=0)
        self.assertTrue(
            bool(jnp.array_equal(led.to_coeffs(jnp.array([3, 1, 6], dtype=KB)), want))
        )

    def test_non_natural_node_sets_to_coeffs(self) -> None:
        # EvalDomain handles any finite node set, not just the naturals: a node at a
        # field inverse (1/2) and a non-consecutive integer set both recover coeffs.
        half = jnp.array(1, dtype=KB) / jnp.array(2, dtype=KB)
        # p(x) = 3 + 5x on {0, 1/2}: p(0)=3, p(1/2)=3 + 5/2.
        deg1 = EvalDomain(jnp.stack([jnp.array(0, dtype=KB), half]))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    deg1.to_coeffs(jnp.stack([jnp.array(3, dtype=KB), 3 + 5 * half])),
                    jnp.array([3, 5], dtype=KB),
                )
            )
        )
        # p(x) = 1 + 2x + 3x² on {0, 2, 4}: p = 1, 17, 57.
        deg2 = EvalDomain(jnp.array([0, 2, 4], dtype=KB))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    deg2.to_coeffs(jnp.array([1, 17, 57], dtype=KB)),
                    jnp.array([1, 2, 3], dtype=KB),
                )
            )
        )

    def test_summand_evals_settable_domain(self) -> None:
        # summand_evals samples the round poly wherever the EvalDomain says — here an
        # explicit finite {0, 2, 4} instead of the default ∞-leading Û. Two linear
        # factors over one variable: a(u)=1+2u, b(u)=2+3u, s(u)=a(u)·b(u).
        stacked = jnp.stack([jnp.array([1, 3], dtype=KB), jnp.array([2, 5], dtype=KB)])
        got = summand_evals(
            stacked, lambda x, y: x * y, EvalDomain(jnp.array([0, 2, 4], dtype=KB))
        )
        want = jnp.array([1 * 2, 5 * 8, 9 * 14], dtype=KB)  # u = 0, 2, 4
        self.assertTrue(bool(jnp.array_equal(got, want)))
        # The default Û domain is just one EvalDomain choice: product_round_poly is
        # summand_evals on uhat_domain.
        self.assertTrue(
            bool(
                jnp.array_equal(
                    product_round_poly(stacked),
                    summand_evals(stacked, lambda x, y: x * y, uhat_domain(2, KB)),
                )
            )
        )

    def test_inf_index_places_infinity(self) -> None:
        # `inf_index` is the index of the ∞ (leading-coeff) sample: 0 first, -1 last.
        # sample and to_coeffs are the reverse-order duals across the two.
        p0, p1 = jnp.array([2], dtype=KB), jnp.array([5], dtype=KB)  # slope 3
        first = EvalDomain(jnp.array([1], dtype=KB), inf_index=0)
        last = EvalDomain(jnp.array([1], dtype=KB), inf_index=-1)
        self.assertTrue(
            bool(jnp.array_equal(last.sample(p0, p1)[::-1], first.sample(p0, p1)))
        )
        # p(x)=3+5x on {0} with ∞ last: values [p(0)=3, ∞=5] → coeffs [3, 5].
        tail = EvalDomain(jnp.array([0], dtype=KB), inf_index=-1)
        self.assertTrue(
            bool(
                jnp.array_equal(
                    tail.to_coeffs(jnp.array([3, 5], dtype=KB)),
                    jnp.array([3, 5], dtype=KB),
                )
            )
        )

    def test_compressed_domain_message(self) -> None:
        # compressed_domain(node) via summand_evals is [s(node), s(∞)] of the
        # 2-factor product — node 0 the [c_0, c_2] form, node 1 the [s(1), s(∞)] form.
        a = jnp.array([1, 2, 3, 4], dtype=KB)
        b = jnp.array([5, 6, 7, 8], dtype=KB)
        stacked = jnp.stack([a, b])
        prod = lambda x, y: x * y  # noqa: E731
        s_inf = jnp.sum((a[2:] - a[:2]) * (b[2:] - b[:2]))
        got0 = summand_evals(stacked, prod, compressed_domain(0, KB))
        self.assertTrue(bool(got0[0] == jnp.sum(a[:2] * b[:2])))  # s(0)
        self.assertTrue(bool(got0[1] == s_inf))  # s(∞)
        got1 = summand_evals(stacked, prod, compressed_domain(1, KB))
        self.assertTrue(bool(got1[0] == jnp.sum(a[2:] * b[2:])))  # s(1)
        self.assertTrue(bool(got1[1] == s_inf))  # s(∞)

    def test_summand_evals_weight_and_lsb(self) -> None:
        # `weight` multiplies each hypercube point before the sum (the eq-weight of
        # an eq-weighted sumcheck); `msb=False` binds the low variable (split_pairs).
        a = jnp.array([1, 2, 3, 4], dtype=KB)
        b = jnp.array([5, 6, 7, 8], dtype=KB)
        stacked = jnp.stack([a, b])
        prod = lambda x, y: x * y  # noqa: E731
        w = jnp.array([2, 3], dtype=KB)
        node0 = EvalDomain(jnp.array([0], dtype=KB))  # single finite node s(0)
        msb = summand_evals(stacked, prod, node0, weight=w)
        self.assertTrue(bool(msb[0] == jnp.sum(w * a[:2] * b[:2])))
        lsb = summand_evals(stacked, prod, node0, weight=w, msb=False)
        self.assertTrue(bool(lsb[0] == jnp.sum(w * a[0::2] * b[0::2])))

    def test_summand_evals_weighted_lsb_is_fusion_ready(self) -> None:
        # The weight= / msb=False branches stay one fused reduction, like the default.
        stacked = jnp.stack([jnp.arange(1, 9, dtype=KB), jnp.arange(9, 17, dtype=KB)])
        w = jnp.array([2, 3, 4, 5], dtype=KB)
        dom = EvalDomain(jnp.array([0], dtype=KB))
        assert_fusion_ready(
            lambda s: summand_evals(s, lambda x, y: x * y, dom, weight=w, msb=False),
            stacked,
            reduces=1,
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
            state = fold(state, r)
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
