# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array, lax, tree_util

from zorch.sumcheck import prover, verifier
from zorch.sumcheck.domain import fold, split_halves, split_pairs
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont


def _product_round(degree: int) -> prover.StandardRound:
    return prover.StandardRound(prover.ProductSummand(degree=degree))


class StandardRoundTest(absltest.TestCase):
    def test_round_poly_degree1_single_mle(self) -> None:
        # degree-1, single MLE: s(0)=sum(P0), s(1)=sum(P1)
        f = rand_field(11, (8,), KB)
        msg = _product_round(1)._round_poly(f[None])
        half = 4
        self.assertEqual(msg.shape, (2,))
        self.assertTrue(bool(msg[0] == fnp.sum(f[:half])))
        self.assertTrue(bool(msg[1] == fnp.sum(f[half:])))

    def test_round_poly_degree2_product(self) -> None:
        # two MLEs, summand=prod, degree 2: s(u) = sum_x' (P0a+u*da)(P0b+u*db)
        a = rand_field(12, (8,), KB)
        b = rand_field(13, (8,), KB)
        msg = _product_round(2)._round_poly(fnp.stack([a, b]))
        self.assertEqual(msg.shape, (3,))
        for u in range(3):
            uf = fnp.array(u, KB)
            fa = a[:4] + uf * (a[4:] - a[:4])
            fb = b[:4] + uf * (b[4:] - b[:4])
            self.assertTrue(bool(msg[u] == fnp.sum(fa * fb)))

    def test_call_threads_state_transcript_msg(self) -> None:
        f = rand_field(15, (8,), KB)
        state, _, msg = _product_round(1)(f[None], cheap_transcript(KB))
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state.shape, (1, 4))  # width halved — one round consumed

    def test_round_poly_is_fusion_ready(self) -> None:
        a = rand_field(16, (8,), KB)
        b = rand_field(17, (8,), KB)
        assert_fusion_ready(_product_round(2)._round_poly, fnp.stack([a, b]), reduces=1)

    def test_degree_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            prover.ProductSummand(degree=0)


class CompressedProductRoundTest(absltest.TestCase):
    def test_message_is_c0_and_leading_coeff(self) -> None:
        # Reconstruct [c0, c1, c2] from the compressed message plus the eval
        # form's s(1); the polynomial must reproduce every eval-form point.
        a = rand_field(30, (8,), KB)
        b = rand_field(31, (8,), KB)
        stacked = fnp.stack([a, b])
        evals = _product_round(2)._round_poly(stacked)  # s(0..2)
        comp = prover.CompressedProductRound()._round_poly(stacked)  # [c0, c2]
        self.assertEqual(comp.shape, (2,))
        self.assertTrue(bool(comp[0] == evals[0]))
        c1 = evals[1] - comp[0] - comp[1]
        for u in range(3):
            u_pt = fnp.array(u, KB)
            want = comp[0] + u_pt * c1 + u_pt * u_pt * comp[1]
            self.assertTrue(bool(want == evals[u]))

    def test_call_threads_state_transcript_msg(self) -> None:
        a = rand_field(32, (8,), KB)
        b = rand_field(33, (8,), KB)
        state, _, msg = prover.CompressedProductRound()(
            fnp.stack([a, b]), cheap_transcript(KB)
        )
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state.shape, (2, 4))  # width halved — one round consumed

    def test_round_poly_is_fusion_ready(self) -> None:
        a = rand_field(34, (8,), KB)
        b = rand_field(35, (8,), KB)
        assert_fusion_ready(
            prover.CompressedProductRound()._round_poly, fnp.stack([a, b]), reduces=1
        )

    def test_rejects_wrong_factor_count(self) -> None:
        f = rand_field(36, (8,), KB)
        with self.assertRaises(ValueError):
            prover.CompressedProductRound()._round_poly(f[None])


class FoldTest(absltest.TestCase):
    def test_split_halves_splits_contiguous_halves(self) -> None:
        f = fnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], KB)
        p0, p1 = split_halves(f)
        self.assertTrue(bool(fnp.all(p0 == fnp.array([[0, 1], [4, 5]], KB))))
        self.assertTrue(bool(fnp.all(p1 == fnp.array([[2, 3], [6, 7]], KB))))

    def test_split_pairs_strides_the_last_axis(self) -> None:
        f = fnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], KB)
        p0, p1 = split_pairs(f)
        self.assertTrue(bool(fnp.all(p0 == fnp.array([[0, 2], [4, 6]], KB))))
        self.assertTrue(bool(fnp.all(p1 == fnp.array([[1, 3], [5, 7]], KB))))

    def test_msb_fold_splits_contiguous_halves(self) -> None:
        f = rand_field(20, (8,), KB)
        r = fnp.array(6, KB)
        got = fold(f, r)
        want = f[:4] + r * (f[4:] - f[:4])
        self.assertEqual(got.shape, (4,))
        self.assertTrue(bool(fnp.all(got == want)))

    def test_lsb_fold_splits_stride_2_pairs(self) -> None:
        f = rand_field(21, (8,), KB)
        r = fnp.array(6, KB)
        got = fold(f, r, msb=False)
        want = f[0::2] + r * (f[1::2] - f[0::2])
        self.assertEqual(got.shape, (4,))
        self.assertTrue(bool(fnp.all(got == want)))

    def test_lsb_fold_is_the_stride_dual_of_msb_fold(self) -> None:
        # An LSB fold on an interleaved buffer equals an MSB fold on the
        # deinterleaved one: same pairs, different layout.
        f = rand_field(22, (8,), KB)
        r = fnp.array(9, KB)
        deinterleaved = fnp.concatenate([f[0::2], f[1::2]])
        self.assertTrue(bool(fnp.all(fold(f, r, msb=False) == fold(deinterleaved, r))))


class SumcheckRoundPytreeTest(absltest.TestCase):
    def test_verifier_config_is_static_pytree(self) -> None:
        # The verifier round is a `verify` scan argument, so degree must be config,
        # not data: zero array leaves, threading through the scan as a static pytree.
        # (The prover round runs under the fold_rounds host loop, never a jit arg, so
        # it is a plain class — no pytree registration.)
        leaves, _ = tree_util.tree_flatten(verifier.SumcheckRound(degree=2))
        self.assertEqual(leaves, [])

    def test_round_msg_survives_scan_as_output(self) -> None:
        # RoundMsg is the per-step output of the fold_rounds / jagged-zerocheck
        # scan, so it must be a registered pytree: its two array fields are data
        # leaves, no meta. A plain frozen dataclass is "not a valid JAX type" as a
        # scan output leaf.
        msg = prover.RoundMsg(round_poly=fnp.zeros(3, KB), challenge=fnp.ones((), KB))
        leaves, _ = tree_util.tree_flatten(msg)
        self.assertEqual(len(leaves), 2)

        # The regression guard: a scan emitting one RoundMsg per step must stack
        # them into RoundMsg(round_poly=(n, ...), challenge=(n, ...)).
        xs = rand_field(7, (4, 3), KB)

        def step(carry: Array, x: Array) -> tuple[Array, prover.RoundMsg]:
            return carry, prover.RoundMsg(round_poly=x, challenge=fnp.sum(x))

        _, msgs = lax.scan(step, fnp.zeros((), KB), xs)
        self.assertEqual(msgs.round_poly.shape, (4, 3))
        self.assertEqual(msgs.challenge.shape, (4,))


if __name__ == "__main__":
    absltest.main()
