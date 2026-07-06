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


class LsbHelpersTest(absltest.TestCase):
    def test_split_pairs_strides_the_last_axis(self) -> None:
        f = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], KB)
        p0, p1 = prover.split_pairs(f)
        self.assertTrue(bool(jnp.all(p0 == jnp.array([[0, 2], [4, 6]], KB))))
        self.assertTrue(bool(jnp.all(p1 == jnp.array([[1, 3], [5, 7]], KB))))

    def test_fold_lsb_matches_manual_pair_fold(self) -> None:
        f = rand_field(21, (8,), KB)
        r = jnp.array(6, KB)
        got = prover.fold_lsb(f, r)
        want = f[0::2] + r * (f[1::2] - f[0::2])
        self.assertEqual(got.shape, (4,))
        self.assertTrue(bool(jnp.all(got == want)))

    def test_fold_lsb_is_the_stride_dual_of_fold(self) -> None:
        # fold_lsb on an interleaved buffer equals `fold` on the deinterleaved
        # one: same pairs, different layout.
        f = rand_field(22, (8,), KB)
        r = jnp.array(9, KB)
        deinterleaved = jnp.concatenate([f[0::2], f[1::2]])
        self.assertTrue(
            bool(jnp.all(prover.fold_lsb(f, r) == prover.fold([deinterleaved], r)[0]))
        )

    def test_zero_extend_pads_the_last_axis(self) -> None:
        f = rand_field(23, (2, 3), KB)
        got = prover.zero_extend(f, 5)
        self.assertEqual(got.shape, (2, 5))
        self.assertTrue(bool(jnp.all(got[:, :3] == f)))
        self.assertTrue(bool(jnp.all(got[:, 3:] == jnp.zeros((2, 2), KB))))

    def test_zero_extend_at_width_is_identity(self) -> None:
        f = rand_field(24, (4,), KB)
        self.assertIs(prover.zero_extend(f, 4), f)

    def test_zero_extend_rejects_shrinking(self) -> None:
        with self.assertRaises(ValueError):
            prover.zero_extend(rand_field(25, (4,), KB), 3)


class SumcheckRoundPytreeTest(absltest.TestCase):
    def test_config_is_static_pytree(self) -> None:
        # degree is config, not data: prover and verifier rounds carry zero array
        # leaves, so each threads through jit/vmap as a fully-static pytree.
        for rnd in (prover.SumcheckRound(degree=2), verifier.SumcheckRound(degree=2)):
            leaves, _ = tree_util.tree_flatten(rnd)
            self.assertEqual(leaves, [])


if __name__ == "__main__":
    absltest.main()
