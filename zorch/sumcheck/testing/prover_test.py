# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, lax, tree_util

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


class CompressedProductRoundTest(absltest.TestCase):
    def test_message_is_c0_and_leading_coeff(self) -> None:
        # Reconstruct [c0, c1, c2] from the compressed message plus the eval
        # form's s(1); the polynomial must reproduce every eval-form point.
        a = rand_field(30, (8,), KB)
        b = rand_field(31, (8,), KB)
        evals = prover.SumcheckRound(degree=2)._round_poly([a, b])  # s(0..2)
        comp = prover.CompressedProductRound()._round_poly([a, b])  # [c0, c2]
        self.assertEqual(comp.shape, (2,))
        self.assertTrue(bool(comp[0] == evals[0]))
        c1 = evals[1] - comp[0] - comp[1]
        for u in range(3):
            u_pt = jnp.array(u, KB)
            want = comp[0] + u_pt * c1 + u_pt * u_pt * comp[1]
            self.assertTrue(bool(want == evals[u]))

    def test_call_threads_state_transcript_msg(self) -> None:
        a = rand_field(32, (8,), KB)
        b = rand_field(33, (8,), KB)
        state, _, msg = prover.CompressedProductRound()([a, b], cheap_transcript(KB))
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state[0].shape, (4,))  # width halved — one round consumed

    def test_round_poly_is_fusion_ready(self) -> None:
        a = rand_field(34, (8,), KB)
        b = rand_field(35, (8,), KB)
        assert_fusion_ready(
            prover.CompressedProductRound()._round_poly, [a, b], reduces=1
        )

    def test_rejects_wrong_factor_count(self) -> None:
        f = rand_field(36, (8,), KB)
        with self.assertRaises(ValueError):
            prover.CompressedProductRound()._round_poly([f])


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


class SumcheckRoundPytreeTest(absltest.TestCase):
    def test_verifier_config_is_static_pytree(self) -> None:
        # The verifier round is a `verify` scan argument, so degree must be config,
        # not data: zero array leaves, threading through the scan as a static pytree.
        # (The prover round runs under the fold_rounds host loop, never a jit arg, so
        # it is a plain frozen dataclass — no pytree registration.)
        leaves, _ = tree_util.tree_flatten(verifier.SumcheckRound(degree=2))
        self.assertEqual(leaves, [])

    def test_round_msg_survives_scan_as_output(self) -> None:
        # RoundMsg is the per-step output of the fold_rounds / jagged-zerocheck
        # scan, so it must be a registered pytree: its two array fields are data
        # leaves, no meta. A plain frozen dataclass is "not a valid JAX type" as a
        # scan output leaf.
        msg = prover.RoundMsg(round_poly=jnp.zeros(3, KB), challenge=jnp.ones((), KB))
        leaves, _ = tree_util.tree_flatten(msg)
        self.assertEqual(len(leaves), 2)

        # The regression guard: a scan emitting one RoundMsg per step must stack
        # them into RoundMsg(round_poly=(n, ...), challenge=(n, ...)).
        xs = rand_field(7, (4, 3), KB)

        def step(carry: Array, x: Array) -> tuple[Array, prover.RoundMsg]:
            return carry, prover.RoundMsg(round_poly=x, challenge=jnp.sum(x))

        _, msgs = lax.scan(step, jnp.zeros((), KB), xs)
        self.assertEqual(msgs.round_poly.shape, (4, 3))
        self.assertEqual(msgs.challenge.shape, (4,))


if __name__ == "__main__":
    absltest.main()
