# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.sumcheck.round import SumcheckRound
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


class SumcheckRoundTest(absltest.TestCase):
    def test_round_poly_degree1_single_mle(self):
        # degree-1, single MLE: s(0)=sum(P0), s(1)=sum(P1)
        f = rand_field(11, (8,), KB)
        rnd = SumcheckRound(degree=1)
        msg = rnd._round_poly([f])
        half = 4
        self.assertEqual(msg.shape, (2,))
        self.assertTrue(bool(msg[0] == jnp.sum(f[:half])))
        self.assertTrue(bool(msg[1] == jnp.sum(f[half:])))

    def test_round_poly_degree2_product(self):
        # two MLEs, combine=prod (default), degree 2: s(u) = sum_x' (P0a+u*da)(P0b+u*db)
        a = rand_field(12, (8,), KB)
        b = rand_field(13, (8,), KB)
        rnd = SumcheckRound(degree=2)
        msg = rnd._round_poly([a, b])
        self.assertEqual(msg.shape, (3,))
        for u in range(3):
            uf = jnp.array(u, KB)
            fa = a[:4] + uf * (a[4:] - a[:4])
            fb = b[:4] + uf * (b[4:] - b[:4])
            self.assertTrue(bool(msg[u] == jnp.sum(fa * fb)))

    def test_fold_matches_manual(self):
        f = rand_field(14, (8,), KB)
        r = jnp.array(6, KB)
        rnd = SumcheckRound(degree=1)
        got = rnd._fold([f], r)[0]
        want = f[:4] + r * (f[4:] - f[:4])
        self.assertTrue(bool(jnp.all(got == want)))

    def test_call_threads_state_transcript_msg(self):
        f = rand_field(15, (8,), KB)
        rnd = SumcheckRound(degree=1)
        t = StubTranscript(jnp.array([3, 0, 0], dtype=KB))
        state, t2, msg = rnd([f], t)
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state[0].shape, (4,))  # width halved
        self.assertEqual(t2.pos, 1)  # one challenge consumed

    def test_round_poly_is_fusion_ready(self):
        a = rand_field(16, (8,), KB)
        b = rand_field(17, (8,), KB)
        assert_fusion_ready(SumcheckRound(degree=2)._round_poly, [a, b], reduces=1)

    def test_degree_must_be_positive(self):
        with self.assertRaises(ValueError):
            SumcheckRound(degree=0)


if __name__ == "__main__":
    absltest.main()
