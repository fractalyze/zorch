# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, tree_util

from zorch.prove import prove
from zorch.sumcheck import prover, verifier
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

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
        t = StubTranscript(jnp.array([3, 0, 0], dtype=KB))
        state, t2, msg = rnd([f], t)
        self.assertEqual(msg.shape, (2,))
        self.assertEqual(state[0].shape, (4,))  # width halved
        if not isinstance(t2, StubTranscript):
            raise AssertionError("expected StubTranscript")
        self.assertEqual(t2.pos, 1)  # one challenge consumed

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


def _composite_eqn(fn: Callable[..., Array], *args: Array) -> Any:
    """The `composite` primitive eqn from `fn`'s jaxpr. Reads the marker envelope
    without lowering to MLIR, so it runs on a jaxlib that lacks
    `stablehlo.CompositeOp` (the released CI build) as well as the
    composite-capable one."""
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    return next(e for e in jaxpr.eqns if e.primitive.name == "composite")


def _composite_attrs(eqn: Any) -> dict[str, Any]:
    """Composite attributes as a plain dict; each scalar flattens to one leaf."""
    return {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}


class ProveCompositeTest(absltest.TestCase):
    """`prove_composite` emits the marker zkx fuses; here, unrecognized, it inlines
    to its body, so it is bit-equal to `prove` on plain JAX (the recognized->fused
    GPU path is exercised e2e separately)."""

    def _assert_inline_equals_prove(
        self, round: prover.SumcheckRound, factors: list[Array], challenges: Array
    ) -> None:
        # prove samples num_vars challenges; the marker takes the num_vars-1 fold
        # challenges (the last round's is never used by a message), both replayed
        # from the same stream, so the messages match.
        got = prover.prove_composite(round, factors, challenges[:-1])
        _, _, msgs = prove(round, list(factors), StubTranscript(challenges))
        want = msgs.round_poly.reshape(-1)  # flat round-major, matches the marker
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_inline_equals_prove_degree2(self) -> None:
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        self._assert_inline_equals_prove(
            prover.SumcheckRound(degree=2), [a, b], rand_field(42, (4,), KB)
        )

    def test_inline_equals_prove_degree1_single_mle(self) -> None:
        f = rand_field(43, (1 << 5,), KB)
        self._assert_inline_equals_prove(
            prover.SumcheckRound(degree=1), [f], rand_field(44, (5,), KB)
        )

    def test_inline_equals_prove_extension_challenges(self) -> None:
        # Base-field factor, extension-field fold challenges (koalabearx4): the
        # round's _round_poly tracks the folded state's dtype across rounds.
        ef = zk_dtypes.koalabearx4_mont
        f = rand_field(45, (1 << 4,), KB).astype(ef)
        self._assert_inline_equals_prove(
            prover.SumcheckRound(degree=1), [f], rand_field(46, (4,), KB).astype(ef)
        )

    def test_operand_layout_has_no_hoisted_const(self) -> None:
        # The fusion ABI is exactly [factor tables][num_vars-1 challenges]; reusing
        # the round's _round_poly must not leak its `us` domain array as an operand.
        a = rand_field(47, (1 << 4,), KB)
        b = rand_field(48, (1 << 4,), KB)
        ch = rand_field(49, (3,), KB)
        eqn = _composite_eqn(
            lambda x, y, c: prover.prove_composite(
                prover.SumcheckRound(degree=2), [x, y], c
            ),
            a,
            b,
            ch,
        )
        self.assertEqual(len(eqn.invars), 2 + 3)  # num_factors + (num_vars-1)

    def test_marker_attributes_match_recognition_contract(self) -> None:
        a = rand_field(50, (1 << 4,), KB)
        b = rand_field(51, (1 << 4,), KB)
        ch = rand_field(52, (3,), KB)
        eqn = _composite_eqn(
            lambda x, y, c: prover.prove_composite(
                prover.SumcheckRound(degree=2),
                [x, y],
                c,
                eq_poly_index=0,
                small_value=True,
            ),
            a,
            b,
            ch,
        )
        self.assertEqual(eqn.params["name"], prover.SUMCHECK_MARKER)
        self.assertEqual(eqn.params["version"], prover.SUMCHECK_MARKER_VERSION)
        self.assertEqual(
            _composite_attrs(eqn),
            {
                "degree": 2,
                "num_vars": 4,
                "num_factors": 2,
                "eq_poly_index": 0,
                "small_value": True,
            },
        )

    def test_default_attributes_are_baseline(self) -> None:
        f = rand_field(53, (1 << 4,), KB)
        ch = rand_field(54, (3,), KB)
        attrs = _composite_attrs(
            _composite_eqn(
                lambda x, c: prover.prove_composite(
                    prover.SumcheckRound(degree=1), [x], c
                ),
                f,
                ch,
            )
        )
        self.assertEqual(attrs["eq_poly_index"], -1)
        self.assertFalse(attrs["small_value"])

    def test_wrong_challenge_count_raises(self) -> None:
        a = rand_field(58, (1 << 4,), KB)
        b = rand_field(59, (1 << 4,), KB)
        ch = rand_field(60, (2,), KB)  # need num_vars-1 = 3, got 2
        with self.assertRaises(ValueError):
            prover.prove_composite(prover.SumcheckRound(degree=2), [a, b], ch)

    def test_empty_factors_raises(self) -> None:
        # Fail loud at the entry, not IndexError on factors[0].
        with self.assertRaises(ValueError):
            prover.prove_composite(
                prover.SumcheckRound(degree=1), [], rand_field(61, (3,), KB)
            )

    def test_non_1d_challenges_raises(self) -> None:
        # challenges must be a flat vector of scalars (the operand contract).
        f = rand_field(62, (1 << 4,), KB)
        ch2d = rand_field(63, (3, 1), KB)
        with self.assertRaises(ValueError):
            prover.prove_composite(prover.SumcheckRound(degree=1), [f], ch2d)


if __name__ == "__main__":
    absltest.main()
