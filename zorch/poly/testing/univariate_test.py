# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array, lax

from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
    eval_univariate,
    intt_with_root,
    ntt_with_root,
    powers,
)
from zorch.testkit.random_field import rand_ext_field, rand_field

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


class InttWithRootTest(absltest.TestCase):
    def test_recovers_coefficients(self) -> None:
        # intt_with_root inverts evaluation on the subgroup <w> (coset_inv=None)
        # and on a coset s*<w> (coset_inv = s^-1): interpolate coefficients from
        # the values and recover the original polynomial byte-for-byte. Oracle is
        # a plain Horner evaluation on the coset (shares no code with the INTT
        # butterfly), batched over columns as fri.fold's final-poly INTT is.
        for k in (2, 4, 8):
            subgroup = _domain(k, KB)  # [w^0..w^{k-1}], order-k root
            omega_inv = fnp.ones((), KB) / subgroup[1]
            coeffs = rand_ext_field(40 + k, (3, k), KB, EF)  # 3 polys, degree <k

            evals = frx.vmap(lambda c: _horner(c, subgroup))(coeffs)  # (3, k)
            self.assertTrue(
                bool(fnp.all(intt_with_root(evals, omega_inv) == coeffs)),
                msg=f"subgroup k={k}",
            )

            shifts = rand_field(50 + k, (3,), KB) + fnp.ones((), KB)  # 3 nonzero
            coset_inv = fnp.ones((3,), KB) / shifts
            points = shifts[:, None] * subgroup[None, :]  # (3, k) coset points
            cevals = frx.vmap(_horner)(coeffs, points)  # (3, k)
            self.assertTrue(
                bool(fnp.all(intt_with_root(cevals, omega_inv, coset_inv) == coeffs)),
                msg=f"coset k={k}",
            )

    def test_rejects_bad_shapes(self) -> None:
        # Fail loud at the public seam: a non-scalar root, or a coset_inv that
        # cannot broadcast against the batch dims, is a caller error.
        groups = rand_ext_field(60, (3, 4), KB, EF)
        omega_inv = fnp.ones((), KB)
        with self.assertRaises(ValueError):
            intt_with_root(groups, fnp.ones((2,), KB))  # non-scalar omega_inv
        with self.assertRaises(ValueError):
            intt_with_root(groups, omega_inv, fnp.ones((5,), KB))  # (5,) vs (3,)
        # Both of these are broadcast-*compatible* with (3,) yet widen the
        # result — (3, 1) to (3, 3, 4) and (2, 1) to (2, 3, 4) — instead of the
        # documented (3, 4). A symmetric compatibility check lets them through.
        for widening in ((3, 1), (2, 1)):
            with self.subTest(coset_inv=widening):
                with self.assertRaises(ValueError):
                    intt_with_root(groups, omega_inv, fnp.ones(widening, KB))
        # A scalar and an exact-match batch both stay (3, 4).
        for ok in (fnp.ones((), KB), fnp.ones((3,), KB)):
            self.assertEqual(intt_with_root(groups, omega_inv, ok).shape, (3, 4))


class NttWithRootTest(absltest.TestCase):
    def test_evaluates_coefficients(self) -> None:
        # Forward dual of intt_with_root: coefficients to values on the subgroup
        # <w> (coset=None) and on a coset s*<w> (coset=s). Same Horner oracle,
        # which shares no code with the butterfly.
        for k in (2, 4, 8):
            subgroup = _domain(k, KB)  # [w^0..w^{k-1}], order-k root
            omega = subgroup[1]
            coeffs = rand_ext_field(70 + k, (3, k), KB, EF)  # 3 polys, degree <k

            want = frx.vmap(lambda c: _horner(c, subgroup))(coeffs)  # (3, k)
            self.assertTrue(
                bool(fnp.all(ntt_with_root(coeffs, omega) == want)),
                msg=f"subgroup k={k}",
            )

            shifts = rand_field(80 + k, (3,), KB) + fnp.ones((), KB)  # 3 nonzero
            points = shifts[:, None] * subgroup[None, :]  # (3, k) coset points
            cwant = frx.vmap(_horner)(coeffs, points)  # (3, k)
            self.assertTrue(
                bool(fnp.all(ntt_with_root(coeffs, omega, shifts) == cwant)),
                msg=f"coset k={k}",
            )

    def test_inverts_intt_with_root(self) -> None:
        # The pair round-trips byte-for-byte in both directions, on the subgroup
        # and on a coset -- the property that makes them duals rather than two
        # unrelated transforms. k=1 is the degenerate no-butterfly case.
        for k in (1, 2, 4, 8):
            subgroup = _domain(k, KB)
            omega = subgroup[1] if k > 1 else fnp.ones((), KB)
            omega_inv = fnp.ones((), KB) / omega
            coeffs = rand_ext_field(90 + k, (3, k), KB, EF)

            evals = ntt_with_root(coeffs, omega)
            self.assertTrue(
                bool(fnp.all(intt_with_root(evals, omega_inv) == coeffs)),
                msg=f"subgroup round trip k={k}",
            )

            shifts = rand_field(100 + k, (3,), KB) + fnp.ones((), KB)
            shifts_inv = fnp.ones((3,), KB) / shifts
            cevals = ntt_with_root(coeffs, omega, shifts)
            self.assertTrue(
                bool(fnp.all(intt_with_root(cevals, omega_inv, shifts_inv) == coeffs)),
                msg=f"coset round trip k={k}",
            )

    def test_rejects_bad_shapes(self) -> None:
        # Same public seam as intt_with_root: a non-power-of-two factor, a
        # non-scalar root, or a coset that cannot broadcast against the batch.
        groups = rand_ext_field(110, (3, 4), KB, EF)
        omega = fnp.ones((), KB)
        with self.assertRaises(ValueError):
            ntt_with_root(rand_ext_field(111, (3, 3), KB, EF), omega)  # k=3 not 2^m
        with self.assertRaises(ValueError):
            ntt_with_root(groups, fnp.ones((2,), KB))  # non-scalar omega
        with self.assertRaises(ValueError):
            ntt_with_root(groups, omega, fnp.ones((5,), KB))  # (5,) vs batch (3,)
        # Broadcast-compatible with (3,) yet widening the result past the
        # documented (3, 4) -- rejected, as in intt_with_root.
        for widening in ((3, 1), (2, 1)):
            with self.subTest(coset=widening):
                with self.assertRaises(ValueError):
                    ntt_with_root(groups, omega, fnp.ones(widening, KB))
        # A scalar and an exact-match batch both stay (3, 4).
        for ok in (fnp.ones((), KB), fnp.ones((3,), KB)):
            self.assertEqual(ntt_with_root(groups, omega, ok).shape, (3, 4))


if __name__ == "__main__":
    absltest.main()
