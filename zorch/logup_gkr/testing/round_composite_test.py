# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""zorch#327 (Z1 prototype): the FS-less `zorch.sumcheck.round` marker around the
dense interaction fold+sum is byte-identical to the eager `_fix_and_sum_int`, and
emits the composite with the pinned operand/attr ABI. This is the marker contract
the xla `SumcheckRecognizer` extension (fractalyze/xla#179) must accept."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.jagged_prover import (
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_sum_as_poly_row,
    _fix_and_sum_int,
    _fix_and_sum_row,
    _InterpConsts,
    _Planes,
    _round_interp_constants,
    _round_metadata,
    _round_poly_row,
    _RoundScalars,
)
from zorch.logup_gkr.testing import random_jagged_layer
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.prover import SUMCHECK_ROUND_MARKER
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear_mont
_PRIME = 2013265921  # koalabear


def _inputs(
    m: int = 8,
) -> tuple[_Planes, jax.Array, jax.Array, _RoundScalars, _InterpConsts]:
    """A random even-width dense-interaction round: four MLE planes, the eq table,
    the previous challenge, the round scalars, and the interp constants."""
    planes = _Planes(*(rand_field(s, (m,), KB) for s in range(4)))
    eq_int = rand_field(10, (m,), KB)
    alpha = rand_field(11, (), KB)
    scalars = _RoundScalars(
        eq_adj=rand_field(12, (), KB),
        pad_adj=rand_field(13, (), KB),
        z_cur=rand_field(14, (), KB),
        claim=rand_field(15, (), KB),
        lam=rand_field(16, (), KB),
    )
    consts = _InterpConsts(*_round_interp_constants(KB))
    return planes, eq_int, alpha, scalars, consts


class RoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        planes, eq_int, alpha, scalars, consts = _inputs()
        want = _fix_and_sum_int(planes, eq_int, alpha, scalars, consts)
        got = _composite_fix_and_sum_dense(planes, eq_int, alpha, scalars, consts)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(bool(jnp.all(g == w)), "marked round diverged from eager")

    def test_emits_marker_with_abi(self) -> None:
        planes, eq_int, alpha, scalars, consts = _inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda p, e, a, s: _composite_fix_and_sum_dense(p, e, a, s, consts)
        )(planes, eq_int, alpha, scalars)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("mid", text)
        self.assertIn("dense", text)


_NRV, _COUNTS = 3, (3, 1, 5, 2)  # 4 segments -> niv=2; the export test's row layout


def _round0_inputs(
    seed: int = 0,
    counts: tuple[int, ...] = _COUNTS,
    nrv: int = _NRV,
) -> tuple[
    _Planes,
    jax.Array | None,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    _RoundScalars,
    _InterpConsts,
]:
    """A valid round-0 jagged row round in the `_round_poly_row` operand order:
    planes, gather, col_index, pair_index, eq_row, eq_int, scalars; `consts`
    last. Built from a random jagged layer + the host round schedule; `gather`
    is None when `counts` is already even (the no-re-pad layout)."""
    consts = _InterpConsts(*_round_interp_constants(KB))
    layer = random_jagged_layer(seed, counts)
    planes0 = _Planes(
        layer.numerator_0, layer.numerator_1, layer.denominator_0, layer.denominator_1
    )
    one = jnp.ones((), KB)
    rng = np.random.default_rng(seed + 99)
    niv = int(np.log2(len(counts)))
    z_row = jnp.asarray(rng.integers(0, _PRIME, (nrv,), np.uint32)).view(KB)
    z_int = jnp.asarray(rng.integers(0, _PRIME, (niv,), np.uint32)).view(KB)
    eq_row = expand_eq_to_hypercube(z_row, one)
    eq_int = expand_eq_to_hypercube(z_int, one)
    scalars0 = _RoundScalars(*(rand_field(seed * 10 + i, (), KB) for i in range(5)))
    gather0, col0, pair0 = _round_metadata(counts, nrv)[0]
    return planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts


def _row_inputs(
    seed: int = 0,
) -> tuple[
    _Planes,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    _RoundScalars,
    _InterpConsts,
]:
    """A valid round-1 jagged row round (the post-round-0 re-padded state), in the
    `_fix_and_sum_row` operand order: planes, eq_row, alpha, gather, col_index,
    pair_index, eq_int, scalars; `consts` closed over. Built by advancing the
    round-0 setup one round, mirroring the export test's round-1 setup so the
    gather/index shapes stay self-consistent over a genuinely jagged size."""
    planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts = _round0_inputs(
        seed
    )
    _poly, planes1 = _round_poly_row(
        planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts
    )
    alpha = rand_field(seed + 7, (), KB)
    gather1, col1, pair1 = _round_metadata(_COUNTS, _NRV)[1]
    scalars1 = _RoundScalars(
        *(rand_field(seed * 10 + 100 + i, (), KB) for i in range(5))
    )
    return planes1, eq_row, alpha, gather1, col1, pair1, eq_int, scalars1, consts


class RowRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        pl, er, al, ga, ci, pi, ei, sc, consts = _row_inputs()
        want = _fix_and_sum_row(pl, er, al, ga, ci, pi, ei, sc, consts)
        got = _composite_fix_and_sum_row(pl, er, al, ga, ci, pi, ei, sc, consts)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(
                bool(jnp.all(g == w)), "marked row round diverged from eager"
            )

    def test_emits_marker_with_abi(self) -> None:
        pl, er, al, ga, ci, pi, ei, sc, consts = _row_inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda pl, er, al, ga, ci, pi, ei, sc: _composite_fix_and_sum_row(
                pl, er, al, ga, ci, pi, ei, sc, consts
            )
        )(pl, er, al, ga, ci, pi, ei, sc)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("mid", text)
        self.assertIn("jagged", text)


class FirstRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        pl, ga, ci, pi, er, ei, sc, consts = _round0_inputs()
        self.assertIsNotNone(ga)  # odd segments -> a real re-pad this round
        want = _round_poly_row(pl, ga, ci, pi, er, ei, sc, consts)
        got = _composite_sum_as_poly_row(pl, ga, ci, pi, er, ei, sc, consts)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(
                bool(jnp.all(g == w)), "marked first round diverged from eager"
            )

    def test_none_gather_byte_identical(self) -> None:
        # An already-even layout: the eager kernel skips the re-pad (`gather`
        # None) while the marker resolves it to a full-height identity gather --
        # a no-op pad, so still byte-identical.
        pl, ga, ci, pi, er, ei, sc, consts = _round0_inputs(counts=(4, 4), nrv=2)
        self.assertIsNone(ga)
        want = _round_poly_row(pl, None, ci, pi, er, ei, sc, consts)
        got = _composite_sum_as_poly_row(pl, None, ci, pi, er, ei, sc, consts)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(
                bool(jnp.all(g == w)), "no-re-pad first round diverged from eager"
            )

    def test_emits_marker_with_abi(self) -> None:
        pl, ga, ci, pi, er, ei, sc, consts = _round0_inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda pl, ga, ci, pi, er, ei, sc: _composite_sum_as_poly_row(
                pl, ga, ci, pi, er, ei, sc, consts
            )
        )(pl, ga, ci, pi, er, ei, sc)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("first", text)
        self.assertIn("jagged", text)


if __name__ == "__main__":
    absltest.main()
