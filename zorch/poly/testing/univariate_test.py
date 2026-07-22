# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array, lax

from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
    eval_univariate,
    powers,
)
from zorch.testkit.random_field import rand_ext_field

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _domain(n: int, dtype: Any) -> Array:
    """The order-n subgroup [w^0, ..., w^{n-1}] for `lax.ntt`'s canonical root,
    recovered independently of any encoder: NTT(e_1)_j = w^j. The transforms
    under test carry no root convention, so this is only a convenient source of
    a genuine order-n root."""
    e1 = fnp.zeros((n,), dtype).at[1].set(fnp.ones((), dtype))
    return lax.ntt(e1, ntt_type="NTT", ntt_length=n)


def _horner(coeffs: Array, points: Array) -> Array:
    """Evaluate the polynomial with `coeffs` at every point in `points`."""
    acc = points * fnp.zeros((), points.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * points + coeffs[i]
    return acc


class PowersTest(absltest.TestCase):
    def test_ascending_powers(self) -> None:
        self.assertEqual([int(v) for v in powers(fnp.array(3, KB), 4)], [1, 3, 9, 27])

    def test_non_power_of_two_length_truncates(self) -> None:
        # Log-doubling overshoots to the next power of two, then truncates.
        self.assertEqual(
            [int(v) for v in powers(fnp.array(2, KB), 5)], [1, 2, 4, 8, 16]
        )

    def test_length_one_is_just_one(self) -> None:
        self.assertEqual([int(v) for v in powers(fnp.array(9, KB), 1)], [1])


class EvalUnivariateTest(absltest.TestCase):
    def test_recovers_node_values(self) -> None:
        evals = fnp.array([3, 5, 11], KB)  # values on [0, 1, 2]
        for i in range(3):
            self.assertTrue(bool(eval_univariate(evals, fnp.array(i, KB)) == evals[i]))

    def test_matches_known_cubic(self) -> None:
        # p(x) = x^3 + 2x + 3 sampled on [0, 1, 2, 3]; check eval off the nodes.
        def p(x: Array) -> Array:
            return x * x * x + fnp.array(2, KB) * x + fnp.array(3, KB)

        evals = fnp.stack([p(fnp.array(i, KB)) for i in range(4)])
        x = fnp.array(7, KB)
        self.assertTrue(bool(eval_univariate(evals, x) == p(x)))


class ComputeLagrangeBasisTest(absltest.TestCase):
    def test_one_hot_at_nodes(self) -> None:
        domain = fnp.array([0, 1, 2, 4, 9], KB)
        for k in range(5):
            basis = compute_lagrange_basis(domain[k], domain)
            for j in range(5):
                want = fnp.array(1 if j == k else 0, KB)
                self.assertTrue(bool(basis[j] == want))

    def test_interpolates_off_nodes(self) -> None:
        # dot(evals, basis) over the integer domain == eval_univariate.
        evals = fnp.array([3, 5, 11, 2], KB)
        domain = fnp.array([0, 1, 2, 3], KB)
        x = fnp.array(7, KB)
        got = fnp.dot(evals, compute_lagrange_basis(x, domain))
        self.assertTrue(bool(got == eval_univariate(evals, x)))


class ComputeInvVandermondeTest(absltest.TestCase):
    def test_recovers_known_quartic_coeffs(self) -> None:
        # p(x) = x^4 + 3x + 7 on {0..4} -> coefficients [7, 3, 0, 0, 1].
        def p(x: Array) -> Array:
            return x * x * x * x + fnp.array(3, KB) * x + fnp.array(7, KB)

        evals = fnp.stack([p(fnp.array(i, KB)) for i in range(5)])
        coeffs = fnp.dot(compute_inv_vandermonde(4, KB), evals)
        self.assertTrue(bool(fnp.all(coeffs == fnp.array([7, 3, 0, 0, 1], KB))))

    def test_ef_evals_promote_at_multiply(self) -> None:
        # The matrix stays base-field; EF evaluations promote in the dot.
        def p(x: Array) -> Array:
            return x * x + fnp.array(2, EF)

        evals = fnp.stack([p(fnp.array(i, EF)) for i in range(3)])
        coeffs = fnp.dot(compute_inv_vandermonde(2, EF), evals)
        self.assertEqual(coeffs.dtype, EF)
        self.assertTrue(bool(fnp.all(coeffs == fnp.array([2, 0, 1], EF))))


class EvalCoeffsTest(absltest.TestCase):
    def test_matches_direct_sum(self) -> None:
        # p(x) = 1 + 2x + 3x^2 at x = 5 -> 86.
        coeffs = fnp.array([1, 2, 3], KB)
        got = eval_coeffs(coeffs, fnp.array(5, KB))
        self.assertTrue(bool(got == fnp.array(86, KB)))

    def test_batched_rows(self) -> None:
        coeffs = fnp.array([[1, 2, 3], [7, 0, 1]], KB)
        got = eval_coeffs(coeffs, fnp.array(2, KB))
        self.assertTrue(bool(fnp.all(got == fnp.array([17, 11], KB))))

    def test_both_schedules_match_the_power_sum(self) -> None:
        # eval_coeffs dispatches on the coefficient count (Horner unroll at or
        # below _HORNER_MAX_COEFFS, prefix-product scan above); both must equal
        # the literal power sum. n spans the threshold; EF batched over rows with
        # a scalar EF point, the fold's shape.
        for n in (1, 8, 9, 33):
            coeffs = rand_ext_field(120 + n, (3, n), KB, EF)
            point = rand_ext_field(130 + n, (), KB, EF)
            want = coeffs[..., 0]
            for i in range(1, n):
                want = want + coeffs[..., i] * point**i
            got = eval_coeffs(coeffs, point)
            self.assertTrue(bool(fnp.all(got == want)), msg=f"n={n}")
            # horner_max pins the schedule; both extremes stay byte-identical.
            for hm in (0, n):
                forced = eval_coeffs(coeffs, point, horner_max=hm)
                self.assertTrue(bool(fnp.all(forced == want)), msg=f"n={n} hm={hm}")


if __name__ == "__main__":
    absltest.main()
