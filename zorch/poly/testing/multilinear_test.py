# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for poly.multilinear: eval_mle, mle_fold, mle_coeffs_to_evals,
mle_evals_to_coeffs (base field)."""
import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.poly.multilinear import (
    _LEVELS,
    _level,
    _mobius_combine,
    _zeta_combine,
    eval_mle,
    mle_coeffs_to_evals,
    mle_evals_to_coeffs,
    mle_fold,
)

F = zk_dtypes.koalabear_mont


def _eval_eq(
    w: Array, x: Array
) -> Array:  # oracle: eq(w,x) = Π (1 - xᵢ - wᵢ + 2 xᵢ wᵢ)
    return fnp.prod(1 - x - w + 2 * x * w)


def _vertex(nat: int, n: int) -> Array:  # nat -> boolean coords, MSB first
    return fnp.array([(nat >> (n - 1 - b)) & 1 for b in range(n)], dtype=F)


class EvalMleTest(absltest.TestCase):
    def test_at_hypercube_vertex_returns_that_eval(self) -> None:
        evals = fnp.arange(8, dtype=F)
        for nat in range(8):
            self.assertTrue(bool(eval_mle(evals, _vertex(nat, 3)) == evals[nat]))

    def test_matches_independent_eq_weighted_sum(self) -> None:
        evals = fnp.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=F)
        x = fnp.array([2, 7, 3], dtype=F)
        want = fnp.zeros((), F)
        for nat in range(8):
            want = want + evals[nat] * _eval_eq(_vertex(nat, 3), x)
        self.assertTrue(bool(eval_mle(evals, x) == want))

    def test_contracts_nonzero_axis(
        self,
    ) -> None:  # 2-D MLE [rows, 2ⁿ], contract axis=1
        mle = fnp.arange(2 * 4, dtype=F).reshape(2, 4)
        x = fnp.array([5, 6], dtype=F)
        out = eval_mle(mle, x, axis=1)
        self.assertEqual(out.shape, (2,))
        for r in range(2):
            self.assertTrue(bool(out[r] == eval_mle(mle[r], x)))


class MleFoldTest(absltest.TestCase):
    def test_pairs_and_halves(self) -> None:
        evals = fnp.arange(8, dtype=F)
        beta = fnp.array(10, dtype=F)
        out = mle_fold(evals, beta)
        self.assertEqual(out.shape, (4,))
        for i in range(4):
            self.assertTrue(bool(out[i] == evals[2 * i] + beta * evals[2 * i + 1]))

    def test_batched_folds_each_row_independently(self) -> None:
        # leading batch axis must ride through (folds the last axis only)
        evals = fnp.arange(2 * 8, dtype=F).reshape(2, 8)
        beta = fnp.array(10, dtype=F)
        out = mle_fold(evals, beta)
        self.assertEqual(out.shape, (2, 4))
        for r in range(2):
            self.assertTrue(bool(fnp.all(out[r] == mle_fold(evals[r], beta))))


class MleCoeffEvalTransformTest(absltest.TestCase):
    def test_coeffs_to_evals_is_subset_sum(self) -> None:
        # eval at vertex v = Σ coeffs[c] over c whose support is a subset of v
        # (the monomial x^c is 1 at v iff c ⊆ v). Small values: no field wrap.
        coeffs = fnp.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=F)
        evals = mle_coeffs_to_evals(coeffs)
        for v in range(8):
            want = sum(int(coeffs[c]) for c in range(8) if (c & v) == c)
            self.assertTrue(bool(evals[v] == fnp.array(want, dtype=F)), msg=f"v={v}")

    def test_roundtrips_both_directions(self) -> None:
        coeffs = fnp.array([7, 2, 9, 0, 1, 8, 3, 4, 6, 5, 2, 1, 9, 0, 4, 7], dtype=F)
        self.assertTrue(
            bool(fnp.all(mle_evals_to_coeffs(mle_coeffs_to_evals(coeffs)) == coeffs))
        )
        evals = fnp.arange(16, dtype=F)
        self.assertTrue(
            bool(fnp.all(mle_coeffs_to_evals(mle_evals_to_coeffs(evals)) == evals))
        )

    def test_evals_agree_with_eval_mle_at_vertices(self) -> None:
        # The produced evals are the MLE on the hypercube: eval_mle at a boolean
        # vertex must return that entry (ties the transform to eval_mle).
        evals = mle_coeffs_to_evals(fnp.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=F))
        for v in range(8):
            self.assertTrue(bool(eval_mle(evals, _vertex(v, 3)) == evals[v]))

    def test_leading_axes_ride_through(self) -> None:
        rows = fnp.arange(2 * 4, dtype=F).reshape(2, 4)
        out = mle_coeffs_to_evals(rows)
        self.assertEqual(out.shape, (2, 4))
        for r in range(2):
            self.assertTrue(bool(fnp.all(out[r] == mle_coeffs_to_evals(rows[r]))))

    def test_degenerate_single_element_is_identity(self) -> None:
        # n = 1 (zero variables): the scan must no-op, not crash on a (…, 2, 0)
        # reshape (lax.scan traces the body even at length 0).
        one = fnp.array([7], dtype=F)
        self.assertTrue(bool(mle_coeffs_to_evals(one)[0] == 7))
        self.assertTrue(bool(mle_evals_to_coeffs(one)[0] == 7))

    def test_lowers_to_a_scan_independent_of_k(self) -> None:
        # The per-bit butterfly is one fixed lax.scan, not a k-deep unroll: the
        # traced graph must carry a `while` and its op count must NOT grow with k
        # (a k-deep unroll would add a kernel per bit — the cost reed_solomon's
        # "no hand-rolled butterfly" note warns about).
        def lowered(k: int) -> str:
            a = fnp.arange(1 << k, dtype=F)
            return frx.jit(mle_evals_to_coeffs).lower(a).as_text()

        small, large = lowered(3), lowered(10)
        self.assertIn("while", small)
        self.assertEqual(small.count("subtract"), large.count("subtract"))


class ButterflyScanBodyIdentityTest(absltest.TestCase):
    """The scan body is memoized, so repeated transforms reuse one trace.

    `lax.scan` keys its trace cache on the identity of the body it is handed, so
    a body built per call re-traces an identical graph every time — invisible in
    results, ~180x the cost of the transform. The wrappers pass module-level
    combines for the same reason: a fresh lambda per call is a fresh key.
    """

    def test_same_combine_and_shape_share_one_body(self) -> None:
        first = _level(_zeta_combine, (), 8)
        second = _level(_zeta_combine, (), 8)

        self.assertIs(first, second)

    def test_shape_and_combine_each_get_their_own_body(self) -> None:
        base = _level(_zeta_combine, (), 8)

        self.assertIsNot(base, _level(_zeta_combine, (), 16))
        self.assertIsNot(base, _level(_zeta_combine, (3,), 8))
        self.assertIsNot(base, _level(_mobius_combine, (), 8))

    def test_wrappers_reuse_one_body_across_calls(self) -> None:
        table = fnp.arange(8, dtype=F)
        before = len(_LEVELS)

        mle_coeffs_to_evals(table)
        mle_coeffs_to_evals(table)
        mle_evals_to_coeffs(table)

        # One entry per (combine, shape), not one per call.
        self.assertLessEqual(len(_LEVELS) - before, 2)


if __name__ == "__main__":
    absltest.main()
