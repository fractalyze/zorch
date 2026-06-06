# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for poly.multilinear: eval_mle, mle_fold (base field)."""
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.multilinear import eval_mle, mle_fold

F = zk_dtypes.koalabear_mont


def _eval_eq(
    w: Array, x: Array
) -> Array:  # oracle: eq(w,x) = Π (1 - xᵢ - wᵢ + 2 xᵢ wᵢ)
    return jnp.prod(1 - x - w + 2 * x * w)


def _vertex(nat: int, n: int) -> Array:  # nat -> boolean coords, MSB first
    return jnp.array([(nat >> (n - 1 - b)) & 1 for b in range(n)], dtype=F)


class EvalMleTest(absltest.TestCase):
    def test_at_hypercube_vertex_returns_that_eval(self) -> None:
        evals = jnp.arange(8, dtype=F)
        for nat in range(8):
            self.assertTrue(bool(eval_mle(evals, _vertex(nat, 3)) == evals[nat]))

    def test_matches_independent_eq_weighted_sum(self) -> None:
        evals = jnp.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=F)
        x = jnp.array([2, 7, 3], dtype=F)
        want = jnp.zeros((), F)
        for nat in range(8):
            want = want + evals[nat] * _eval_eq(_vertex(nat, 3), x)
        self.assertTrue(bool(eval_mle(evals, x) == want))

    def test_contracts_nonzero_axis(
        self,
    ) -> None:  # 2-D MLE [rows, 2ⁿ], contract axis=1
        mle = jnp.arange(2 * 4, dtype=F).reshape(2, 4)
        x = jnp.array([5, 6], dtype=F)
        out = eval_mle(mle, x, axis=1)
        self.assertEqual(out.shape, (2,))
        for r in range(2):
            self.assertTrue(bool(out[r] == eval_mle(mle[r], x)))


class MleFoldTest(absltest.TestCase):
    def test_pairs_and_halves(self) -> None:
        evals = jnp.arange(8, dtype=F)
        beta = jnp.array(10, dtype=F)
        out = mle_fold(evals, beta)
        self.assertEqual(out.shape, (4,))
        for i in range(4):
            self.assertTrue(bool(out[i] == evals[2 * i] + beta * evals[2 * i + 1]))

    def test_batched_folds_each_row_independently(self) -> None:
        # leading batch axis must ride through (folds the last axis only)
        evals = jnp.arange(2 * 8, dtype=F).reshape(2, 8)
        beta = jnp.array(10, dtype=F)
        out = mle_fold(evals, beta)
        self.assertEqual(out.shape, (2, 4))
        for r in range(2):
            self.assertTrue(bool(jnp.all(out[r] == mle_fold(evals[r], beta))))


if __name__ == "__main__":
    absltest.main()
