# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    dot_unrolled,
    eval_coeffs,
    eval_univariate,
)

KB = zk_dtypes.koalabear_mont


class DotUnrolledTest(absltest.TestCase):
    def test_matches_matmul(self) -> None:
        # The dot-free contraction must equal `a @ v` exactly (field
        # arithmetic is exact, so the unroll order is byte-neutral).
        a = jnp.array([3, 1, 4, 1], KB)
        v = jnp.array([5, 9, 2, 6], KB)
        self.assertTrue(bool(dot_unrolled(a, v) == jnp.dot(a, v)))

    def test_contracts_last_axis_of_a(self) -> None:
        # Batched: `a` is (rows, len), contracted against `v` of (len,).
        a = jnp.array([[3, 1], [4, 1], [5, 9]], KB)
        v = jnp.array([2, 6], KB)
        got = dot_unrolled(a, v)
        self.assertEqual(got.shape, (3,))
        self.assertTrue(bool(jnp.all(got == a @ v)))


class EvalUnivariateTest(absltest.TestCase):
    def test_recovers_node_values(self) -> None:
        evals = jnp.array([3, 5, 11], KB)  # values on [0, 1, 2]
        for i in range(3):
            self.assertTrue(bool(eval_univariate(evals, jnp.array(i, KB)) == evals[i]))

    def test_matches_known_cubic(self) -> None:
        # p(x) = x^3 + 2x + 3 sampled on [0, 1, 2, 3]; check eval off the nodes.
        def p(x: Array) -> Array:
            return x * x * x + jnp.array(2, KB) * x + jnp.array(3, KB)

        evals = jnp.stack([p(jnp.array(i, KB)) for i in range(4)])
        x = jnp.array(7, KB)
        self.assertTrue(bool(eval_univariate(evals, x) == p(x)))


class ComputeLagrangeBasisTest(absltest.TestCase):
    def test_one_hot_at_nodes(self) -> None:
        domain = jnp.array([0, 1, 2, 4, 9], KB)
        for k in range(5):
            basis = compute_lagrange_basis(domain[k], domain)
            for j in range(5):
                want = jnp.array(1 if j == k else 0, KB)
                self.assertTrue(bool(basis[j] == want))

    def test_interpolates_off_nodes(self) -> None:
        # dot(evals, basis) over the integer domain == eval_univariate.
        evals = jnp.array([3, 5, 11, 2], KB)
        domain = jnp.array([0, 1, 2, 3], KB)
        x = jnp.array(7, KB)
        got = jnp.dot(evals, compute_lagrange_basis(x, domain))
        self.assertTrue(bool(got == eval_univariate(evals, x)))


class ComputeInvVandermondeTest(absltest.TestCase):
    def test_recovers_known_quartic_coeffs(self) -> None:
        # p(x) = x^4 + 3x + 7 on {0..4} -> coefficients [7, 3, 0, 0, 1].
        def p(x: Array) -> Array:
            return x * x * x * x + jnp.array(3, KB) * x + jnp.array(7, KB)

        evals = jnp.stack([p(jnp.array(i, KB)) for i in range(5)])
        coeffs = jnp.dot(compute_inv_vandermonde(4, KB), evals)
        self.assertTrue(bool(jnp.all(coeffs == jnp.array([7, 3, 0, 0, 1], KB))))

    def test_ef_evals_promote_at_multiply(self) -> None:
        # The matrix stays base-field; EF evaluations promote in the dot.
        EF = zk_dtypes.koalabearx4_mont

        def p(x: Array) -> Array:
            return x * x + jnp.array(2, EF)

        evals = jnp.stack([p(jnp.array(i, EF)) for i in range(3)])
        coeffs = jnp.dot(compute_inv_vandermonde(2, EF), evals)
        self.assertEqual(coeffs.dtype, EF)
        self.assertTrue(bool(jnp.all(coeffs == jnp.array([2, 0, 1], EF))))


class EvalCoeffsTest(absltest.TestCase):
    def test_matches_direct_sum(self) -> None:
        # p(x) = 1 + 2x + 3x^2 at x = 5 -> 86.
        coeffs = jnp.array([1, 2, 3], KB)
        got = eval_coeffs(coeffs, jnp.array(5, KB))
        self.assertTrue(bool(got == jnp.array(86, KB)))

    def test_batched_rows(self) -> None:
        coeffs = jnp.array([[1, 2, 3], [7, 0, 1]], KB)
        got = eval_coeffs(coeffs, jnp.array(2, KB))
        self.assertTrue(bool(jnp.all(got == jnp.array([17, 11], KB))))


if __name__ == "__main__":
    absltest.main()
