# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import tree_util

from zorch.sumcheck import prover, verifier
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


class SumcheckRoundTest(absltest.TestCase):
    def test_round_poly_degree1_single_mle(self) -> None:
        # degree-1, single MLE: s(0)=sum(P0), s(1)=sum(P1)
        f = rand_field(11, (8,), KB)
        rnd = prover.SumcheckRound(degree=1)
        msg = rnd._round_poly([f])
        half = 4
        self.assertEqual(msg.shape, (2,))
        self.assertTrue(bool(msg[0] == jnp.sum(f[:half])))
        self.assertTrue(bool(msg[1] == jnp.sum(f[half:])))

    def test_round_poly_degree2_product(self) -> None:
        # two MLEs, summand=prod, degree 2: s(u) = sum_x' (P0a+u*da)(P0b+u*db)
        a = rand_field(12, (8,), KB)
        b = rand_field(13, (8,), KB)
        rnd = prover.SumcheckRound(degree=2)
        msg = rnd._round_poly([a, b])
        self.assertEqual(msg.shape, (3,))
        for u in range(3):
            uf = jnp.array(u, KB)
            fa = a[:4] + uf * (a[4:] - a[:4])
            fb = b[:4] + uf * (b[4:] - b[:4])
            self.assertTrue(bool(msg[u] == jnp.sum(fa * fb)))

    def test_round_poly_with_batch_dimension(self) -> None:
        # Leading batch dims must broadcast: msg is (degree+1, *batch).
        batch = 3
        a = rand_field(18, (batch, 8), KB)
        b = rand_field(19, (batch, 8), KB)
        msg = prover.SumcheckRound(degree=2)._round_poly([a, b])
        self.assertEqual(msg.shape, (3, batch))
        for u in range(3):
            uf = jnp.array(u, KB)
            fa = a[:, :4] + uf * (a[:, 4:] - a[:, :4])
            fb = b[:, :4] + uf * (b[:, 4:] - b[:, :4])
            self.assertTrue(bool(jnp.all(msg[u] == jnp.sum(fa * fb, axis=-1))))

    def test_fold_matches_manual(self) -> None:
        f = rand_field(14, (8,), KB)
        r = jnp.array(6, KB)
        got = prover.fold([f], r)[0]
        want = f[:4] + r * (f[4:] - f[:4])
        self.assertTrue(bool(jnp.all(got == want)))

    def test_call_threads_state_transcript_msg(self) -> None:
        f = rand_field(15, (8,), KB)
        rnd = prover.SumcheckRound(degree=1)
        state, _, msg = rnd([f], cheap_transcript(KB))
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state[0].shape, (4,))  # width halved — one round consumed

    def test_round_poly_is_fusion_ready(self) -> None:
        a = rand_field(16, (8,), KB)
        b = rand_field(17, (8,), KB)
        assert_fusion_ready(
            prover.SumcheckRound(degree=2)._round_poly, [a, b], reduces=1
        )

    def test_degree_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            prover.SumcheckRound(degree=0)

    def test_split_rejects_odd_width(self) -> None:
        # Fail loud instead of dropping the odd element on `// 2`.
        r = jnp.array(1, KB)
        with self.assertRaises(ValueError):
            prover.fold([rand_field(1, (7,), KB)], r)

    def test_split_rejects_mismatched_shapes(self) -> None:
        a = rand_field(1, (8,), KB)
        b = rand_field(2, (4,), KB)
        with self.assertRaises(ValueError):
            prover.SumcheckRound(degree=2)._round_poly([a, b])


class SumcheckRoundPytreeTest(absltest.TestCase):
    def test_config_is_static_pytree(self) -> None:
        # degree is config, not data: prover and verifier rounds carry zero array
        # leaves, so each threads through jit/vmap as a fully-static pytree.
        for rnd in (prover.SumcheckRound(degree=2), verifier.SumcheckRound(degree=2)):
            leaves, _ = tree_util.tree_flatten(rnd)
            self.assertEqual(leaves, [])


if __name__ == "__main__":
    absltest.main()
