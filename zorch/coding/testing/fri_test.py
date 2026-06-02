# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for coding.fri: fold formula, evaluation domain, FRI commute invariant."""
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.coding.fri import eval_domain, fri_fold, fri_fold_values
from zorch.coding.reed_solomon import ReedSolomon
from zorch.testkit.random_field import rand_field

F = zk_dtypes.koalabear


class FriValuesTest(absltest.TestCase):
    def test_eval_domain_is_negation_symmetric(self) -> None:
        n = 8
        d = eval_domain(F, n)  # [ω⁰..ω⁷]
        self.assertEqual(d.shape, (n,))
        half = n // 2
        # ω^{j+half} = -ω^j  (second half negates the first)
        self.assertTrue(bool(jnp.all(d[half:] + d[:half] == jnp.zeros(half, F))))

    def test_eval_domain_order_one_is_unity(self) -> None:  # trivial subgroup {1}
        self.assertTrue(bool(jnp.all(eval_domain(F, 1) == jnp.ones(1, F))))

    def test_eval_domain_rejects_non_power_of_two(self) -> None:
        with self.assertRaises(ValueError):
            eval_domain(F, 6)

    def test_fold_values_recovers_linear_poly(self) -> None:
        # Independent oracle: for f(X)=a+bX, f(x)=a+bx and f(-x)=a-bx, so the
        # conjugate fold must be exactly a + β·b — no division to hand-compute,
        # and it shares no expression with fri_fold_values.
        a, b, x, beta = (jnp.array(v, dtype=F) for v in (3, 5, 4, 6))
        fx = a + b * x
        fnx = a - b * x
        got = fri_fold_values(fx, fnx, beta, x)
        self.assertTrue(bool(got == a + beta * b))


class FriFoldCommuteTest(absltest.TestCase):
    def test_fold_encode_commute(self) -> None:
        # fold(encode(p), β) == encode(p_even + β·p_odd) on the squared domain.
        k = 4
        L = 1 << k
        p = rand_field(40, (L,), F)  # polynomial coefficients
        beta = rand_field(41, (), F)
        cw = ReedSolomon(L, 1, F).encode(p)  # blowup=1: DFT on order-L domain
        folded = fri_fold(cw, beta)  # (L/2,)
        p_fold = p[0::2] + beta * p[1::2]  # even + β·odd
        expected = ReedSolomon(L // 2, 1, F).encode(p_fold)
        self.assertEqual(folded.shape, (L // 2,))
        self.assertTrue(bool(jnp.all(folded == expected)))


if __name__ == "__main__":
    absltest.main()
