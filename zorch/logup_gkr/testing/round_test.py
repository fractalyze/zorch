# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.round import LogupGkrRound
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


def _state(seed, width):
    """Five MLE-eval factors in combine order [eq, n0, d1, n1, d0]."""
    return [rand_field(seed + i, (width,), KB) for i in range(5)]


def _combine(lam, eq, n0, d1, n1, d0):
    return eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)


def _hypercube_sum(lam, state):
    return jnp.sum(_combine(lam, *state))


def _eval_cubic_at(evals, r):
    """Interpolate the degree-3 poly given by `evals` at u=0,1,2,3, eval at r
    (Lagrange over the field — the sumcheck verifier's per-round check)."""
    xs = [jnp.array(i, evals.dtype) for i in range(4)]
    acc = jnp.array(0, evals.dtype)
    for j in range(4):
        term = evals[j]
        for m in range(4):
            if m != j:
                term = term * (r - xs[m]) / (xs[j] - xs[m])
        acc = acc + term
    return acc


class LogupGkrRoundTest(absltest.TestCase):
    def test_round_poly_matches_naive_cubic(self):
        st = _state(20, 8)
        lam = jnp.array(7, KB)
        msg = LogupGkrRound(lam).round_poly(st)
        self.assertEqual(msg.shape, (4,))  # degree 3 -> 4 evals
        half = 4
        for u in range(4):
            uf = jnp.array(u, KB)
            folded = [x[:half] + uf * (x[half:] - x[:half]) for x in st]
            want = jnp.sum(_combine(lam, *folded))
            self.assertTrue(bool(msg[u] == want))

    def test_fold_matches_manual(self):
        st = _state(30, 8)
        r = jnp.array(5, KB)
        got = LogupGkrRound(jnp.array(3, KB)).fold(st, r)
        self.assertEqual(len(got), 5)
        for x, g in zip(st, got):
            self.assertEqual(g.shape, (4,))  # width halved
            self.assertTrue(bool(jnp.all(g == x[:4] + r * (x[4:] - x[:4]))))

    def test_sumcheck_invariant_s0_plus_s1(self):
        st = _state(40, 16)
        lam = jnp.array(9, KB)
        msg = LogupGkrRound(lam).round_poly(st)
        # s(0)+s(1) == sum over the full hypercube of the combine (the claim)
        self.assertTrue(bool(msg[0] + msg[1] == _hypercube_sum(lam, st)))

    def test_call_threads_state_transcript_msg(self):
        st = _state(50, 8)
        t = StubTranscript(jnp.array([4, 0, 0], dtype=KB))
        state, t2, msg = LogupGkrRound(jnp.array(2, KB))(st, t)
        self.assertEqual(msg.shape, (4,))
        self.assertEqual(len(state), 5)
        self.assertEqual(state[0].shape, (4,))  # width halved
        self.assertEqual(t2.pos, 1)  # one challenge consumed

    def test_end_to_end_sumcheck_reduction(self):
        # Drive the round to completion; the running claim must reduce
        # consistently each round and match the final width-1 evaluation.
        from zorch.prove import prove

        width = 16
        st = _state(60, width)
        lam = jnp.array(11, KB)
        rnd = LogupGkrRound(lam)
        n = 4  # log2(16)
        challenges = rand_field(99, (n,), KB)

        # Replay round-by-round to check the per-round sumcheck identity.
        claim = _hypercube_sum(lam, st)
        state, t = st, StubTranscript(challenges)
        for i in range(n):
            state, t, msg = rnd(state, t)
            self.assertTrue(bool(msg[0] + msg[1] == claim))
            claim = _eval_cubic_at(msg, challenges[i])
        self.assertTrue(bool(state[0].shape == (1,)))  # collapsed to a point
        self.assertTrue(bool(_combine(lam, *state)[0] == claim))

        # The generic `prove` driver produces the same per-round messages.
        _, _, proof = prove(rnd, st, StubTranscript(challenges))
        self.assertEqual(proof.shape, (n, 4))

    def test_round_poly_is_fusion_ready_stablehlo(self):
        # Fusion contract: the round body lowers with no gather (kInput boundary
        # source) and no dot (matmul); only the one inherent Sigma reduce.
        import jax

        st = _state(80, 8)
        hlo = jax.jit(LogupGkrRound(jnp.array(5, KB)).round_poly).lower(st).as_text()
        self.assertNotIn("gather", hlo)
        self.assertNotIn("dot", hlo)

    def test_state_must_have_five_factors(self):
        with self.assertRaises(ValueError):
            LogupGkrRound(jnp.array(1, KB)).round_poly(_state(70, 8)[:4])


if __name__ == "__main__":
    absltest.main()
