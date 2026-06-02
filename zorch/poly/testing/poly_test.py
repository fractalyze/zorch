# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly import eval_univariate

KB = zk_dtypes.koalabear


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


if __name__ == "__main__":
    absltest.main()
