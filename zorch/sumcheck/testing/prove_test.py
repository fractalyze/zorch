# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.prove import prove
from zorch.sumcheck import SumcheckRound
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


def _product(factors):
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _eval_mle(evals, point):
    """Test oracle: evaluate a multilinear (2^n evals) at `point` (MSB-first)."""
    cur = evals
    for r in point:
        half = cur.shape[-1] // 2
        cur = cur[..., :half] + r * (cur[..., half:] - cur[..., :half])
    return cur[0]


def _eval_univariate(evals, x):
    """Test oracle: evaluate a univariate given by its values on
    [0, 1, ..., len-1] at `x` via Lagrange interpolation (small domain)."""
    d = evals.shape[0]
    nodes = jnp.arange(d, dtype=KB).astype(evals.dtype)  # arange(EF) is unsupported
    acc = jnp.zeros((), evals.dtype)
    for i in range(d):
        num = jnp.ones((), evals.dtype)
        den = jnp.ones((), evals.dtype)
        for j in range(d):
            if j != i:
                num = num * (x - nodes[j])
                den = den * (nodes[i] - nodes[j])
        acc = acc + evals[i] * (num / den)
    return acc


class SumcheckProveTest(absltest.TestCase):
    def _check_identity(self, factors, degree, n, seed):
        challenges = rand_field(seed, (n,), KB)
        final_state, _, proof = prove(
            SumcheckRound(degree=degree), list(factors), StubTranscript(challenges)
        )

        # claimed sum == s_0(0) + s_0(1)
        claimed = jnp.sum(_product(list(factors)))
        self.assertTrue(bool(claimed == proof[0][0] + proof[0][1]))

        # round-to-round: s_i(0)+s_i(1) == s_{i-1}(r_{i-1})
        for i in range(1, n):
            lhs = proof[i][0] + proof[i][1]
            rhs = _eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))

        # final fold == product of the MLEs evaluated at the challenge point
        want = _product([_eval_mle(f, challenges) for f in factors])
        self.assertTrue(bool(_product([s[0] for s in final_state]) == want))
        last = _eval_univariate(proof[n - 1], challenges[n - 1])
        self.assertTrue(bool(last == want))

    def test_empty_state_raises(self):
        with self.assertRaises(ValueError):
            prove(SumcheckRound(degree=1), [], StubTranscript(jnp.zeros(1, KB)))

    def test_degree1_single_mle(self):
        f = rand_field(20, (1 << 4,), KB)
        self._check_identity((f,), degree=1, n=4, seed=21)

    def test_degree2_product_two_mles(self):
        a = rand_field(22, (1 << 4,), KB)
        b = rand_field(23, (1 << 4,), KB)
        self._check_identity((a, b), degree=2, n=4, seed=24)

    def test_degree1_extension_challenges(self):
        EF = zk_dtypes.koalabearx4
        f = rand_field(30, (1 << 4,), KB).astype(EF)
        challenges = rand_field(31, (4,), KB).astype(EF)
        final_state, _, proof = prove(
            SumcheckRound(degree=1), [f], StubTranscript(challenges)
        )
        self.assertTrue(bool(jnp.sum(f) == proof[0][0] + proof[0][1]))
        for i in range(1, 4):
            lhs = proof[i][0] + proof[i][1]
            rhs = _eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))
        self.assertTrue(bool(final_state[0][0] == _eval_mle(f, challenges)))


if __name__ == "__main__":
    absltest.main()
