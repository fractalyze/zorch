# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.univariate import eval_univariate
from zorch.prove import prove
from zorch.sumcheck import prover
from zorch.sumcheck.testing import eval_mle_oracle, product
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear
_GPU_BACKEND = jax.default_backend() == "gpu"


class SumcheckProveTest(absltest.TestCase):
    def _check_identity(
        self, factors: Sequence[Array], degree: int, n: int, seed: int
    ) -> None:
        challenges = rand_field(seed, (n,), KB)
        final_state, _, proof = prove(
            prover.SumcheckRound(degree=degree),
            list(factors),
            StubTranscript(challenges),
        )

        # claimed sum == s_0(0) + s_0(1)
        claimed = jnp.sum(product(list(factors)))
        self.assertTrue(bool(claimed == proof[0][0] + proof[0][1]))

        # round-to-round: s_i(0)+s_i(1) == s_{i-1}(r_{i-1})
        for i in range(1, n):
            lhs = proof[i][0] + proof[i][1]
            rhs = eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))

        # final fold == product of the MLEs evaluated at the challenge point
        want = product([eval_mle_oracle(f, challenges) for f in factors])
        self.assertTrue(bool(product([s[0] for s in final_state]) == want))
        last = eval_univariate(proof[n - 1], challenges[n - 1])
        self.assertTrue(bool(last == want))

    def test_empty_state_raises(self) -> None:
        with self.assertRaises(ValueError):
            prove(prover.SumcheckRound(degree=1), [], StubTranscript(jnp.zeros(1, KB)))

    def test_degree1_single_mle(self) -> None:
        f = rand_field(20, (1 << 4,), KB)
        self._check_identity((f,), degree=1, n=4, seed=21)

    def test_degree2_product_two_mles(self) -> None:
        a = rand_field(22, (1 << 4,), KB)
        b = rand_field(23, (1 << 4,), KB)
        self._check_identity((a, b), degree=2, n=4, seed=24)

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_degree1_extension_challenges(self) -> None:
        EF = zk_dtypes.koalabearx4
        f = rand_field(30, (1 << 4,), KB).astype(EF)
        challenges = rand_field(31, (4,), KB).astype(EF)
        final_state, _, proof = prove(
            prover.SumcheckRound(degree=1), [f], StubTranscript(challenges)
        )
        self.assertTrue(bool(jnp.sum(f) == proof[0][0] + proof[0][1]))
        for i in range(1, 4):
            lhs = proof[i][0] + proof[i][1]
            rhs = eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))
        self.assertTrue(bool(final_state[0][0] == eval_mle_oracle(f, challenges)))


if __name__ == "__main__":
    absltest.main()
