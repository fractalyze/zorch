# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""zorch#327 (Z1 prototype): the FS-less `zorch.sumcheck.round` marker around the
dense interaction fold+sum is byte-identical to the eager `_fix_and_sum_int`, and
emits the composite with the pinned operand/attr ABI. This is the marker contract
the xla `SumcheckRecognizer` extension (fractalyze/xla#179) must accept."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_fix_last,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_rounds import (
    _fix_and_sum_boundary,
    _fix_and_sum_int,
    _fix_and_sum_row,
    _fix_last,
    _round_interp_constants,
    _round_poly_row,
)
from zorch.logup_gkr._jagged_schedule import (
    _round_live_meta,
    _round_metadata,
    _round_out_pairs,
    _row_counts_operand,
)
from zorch.logup_gkr._jagged_types import (
    _InterpConsts,
    _Planes,
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
) -> tuple[_Planes, jax.Array, jax.Array, _RoundScalars, _InterpConsts, jax.Array]:
    """A random even-width dense-interaction round: four MLE planes, the eq table,
    the previous challenge, the round scalars, the interp constants, and the
    fully-live `live` prefix marker (`m // 4` reduce pairs)."""
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
    live = jnp.asarray([m // 4, 0], jnp.int32)
    return planes, eq_int, alpha, scalars, consts, live


def _assert_prefix(
    test: absltest.TestCase, got: jax.Array, want: jax.Array, what: str
) -> None:
    """`got`'s live prefix matches `want` exactly and its tail is zeros -- the
    width-preserving round buffer convention (the tail is dead, masked by the
    `live` operand; the decomposition writes it as zeros)."""
    n = want.shape[0]
    test.assertTrue(bool(jnp.all(got[:n] == want)), f"{what} live prefix diverged")
    tail = got[n:]
    test.assertTrue(
        bool(jnp.all(tail == jnp.zeros_like(tail))), f"{what} tail not zeros"
    )


class RoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        planes, eq_int, alpha, scalars, consts, live = _inputs()
        want_poly, want_planes, want_eq = _fix_and_sum_int(
            planes, eq_int, alpha, scalars, consts
        )
        got_poly, got_planes, got_eq = _composite_fix_and_sum_dense(
            planes, eq_int, alpha, scalars, consts, live
        )
        self.assertTrue(
            bool(jnp.all(got_poly == want_poly)), "marked round poly diverged"
        )
        # Width-preserving: the folded state returns at the input width, live
        # prefix byte-identical to the eager halved state, zero tail.
        for name in ("n0", "n1", "d0", "d1"):
            _assert_prefix(
                self, getattr(got_planes, name), getattr(want_planes, name), name
            )
        _assert_prefix(self, got_eq, want_eq, "eq_int")

    def test_partially_live_matches_eager_on_prefix(self) -> None:
        # The fixed-width layout: a 16-wide buffer whose live prefix is the
        # 8-wide round above (live = 2 pairs). The marked round must reproduce
        # the exact-width eager round on the live prefix -- the sum masks the
        # dead tail, the fold rides it as zeros.
        planes, eq_int, alpha, scalars, consts, live = _inputs()
        wide_planes = _Planes(
            *(
                jnp.concatenate([a, jnp.zeros((8,), a.dtype)])
                for a in (planes.n0, planes.n1, planes.d0, planes.d1)
            )
        )
        wide_eq = jnp.concatenate([eq_int, jnp.zeros((8,), eq_int.dtype)])
        want_poly, want_planes, want_eq = _fix_and_sum_int(
            planes, eq_int, alpha, scalars, consts
        )
        got_poly, got_planes, got_eq = _composite_fix_and_sum_dense(
            wide_planes, wide_eq, alpha, scalars, consts, live
        )
        self.assertTrue(
            bool(jnp.all(got_poly == want_poly)), "padded round poly diverged"
        )
        for name in ("n0", "n1", "d0", "d1"):
            _assert_prefix(
                self, getattr(got_planes, name), getattr(want_planes, name), name
            )
        _assert_prefix(self, got_eq, want_eq, "eq_int")

    def test_emits_marker_with_abi(self) -> None:
        planes, eq_int, alpha, scalars, consts, live = _inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda p, e, a, s, lv: _composite_fix_and_sum_dense(p, e, a, s, consts, lv)
        )(planes, eq_int, alpha, scalars, live)
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
    jax.Array,
]:
    """A valid round-0 jagged row round in the `_round_poly_row` operand order:
    planes, gather, col_index, pair_index, eq_row, eq_int, scalars; `consts`
    then the round's `live` marker last. Built from a random jagged layer + the
    host round schedule; `gather` is None when `counts` is already even (the
    no-re-pad layout)."""
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
    gather0, col0, pair0, _live0 = _round_metadata(counts, nrv)[0]
    # The v2 marker operands (row_counts, live triple, static padded width) --
    # the explicit arrays above feed only the eager want-side.
    marker0 = (
        _row_counts_operand(counts),
        _round_live_meta(counts, nrv)[0],
        _round_out_pairs(counts, nrv)[0],
    )
    return planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts, marker0


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
    jax.Array,
]:
    """A valid round-1 jagged row round (the post-round-0 re-padded state), in the
    `_fix_and_sum_row` operand order: planes, eq_row, alpha, gather, col_index,
    pair_index, eq_int, scalars; `consts` then the round's `live` marker last.
    Built by advancing the round-0 setup one round, mirroring the export test's
    round-1 setup so the gather/index shapes stay self-consistent over a
    genuinely jagged size."""
    planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts, _marker0 = (
        _round0_inputs(seed)
    )
    _poly, planes1 = _round_poly_row(
        planes0, gather0, col0, pair0, eq_row, eq_int, scalars0, consts
    )
    alpha = rand_field(seed + 7, (), KB)
    gather1, col1, pair1, _live1 = _round_metadata(_COUNTS, _NRV)[1]
    scalars1 = _RoundScalars(
        *(rand_field(seed * 10 + 100 + i, (), KB) for i in range(5))
    )
    marker1 = (
        _row_counts_operand(_COUNTS),
        _round_live_meta(_COUNTS, _NRV)[1],
        _round_out_pairs(_COUNTS, _NRV)[1],
    )
    return (
        planes1,
        eq_row,
        alpha,
        gather1,
        col1,
        pair1,
        eq_int,
        scalars1,
        consts,
        marker1,
    )


class RowRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        pl, er, al, ga, ci, pi, ei, sc, consts, (rc, live, op) = _row_inputs()
        want_poly, want_planes, want_eq = _fix_and_sum_row(
            pl, er, al, ga, ci, pi, ei, sc, consts
        )
        got_poly, got_planes, got_eq = _composite_fix_and_sum_row(
            pl, er, al, rc, ei, sc, consts, live, op
        )
        self.assertTrue(
            bool(jnp.all(got_poly == want_poly)), "marked row poly diverged"
        )
        # The re-padded planes share the gather width on both paths; eq_row
        # returns width-preserved (folded live prefix, zero tail) against the
        # eager path's halved table.
        for name in ("n0", "n1", "d0", "d1"):
            g, w = getattr(got_planes, name), getattr(want_planes, name)
            self.assertTrue(bool(jnp.all(g == w)), f"row {name} diverged")
        _assert_prefix(self, got_eq, want_eq, "eq_row")

    def test_emits_marker_with_abi(self) -> None:
        pl, er, al, _ga, _ci, _pi, ei, sc, consts, (rc, live, op) = _row_inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda pl, er, al, rc, ei, sc, lv: _composite_fix_and_sum_row(
                pl, er, al, rc, ei, sc, consts, lv, op
            )
        )(pl, er, al, rc, ei, sc, live)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("mid", text)
        self.assertIn("jagged", text)


class RoundClaimStatusTest(absltest.TestCase):
    """The installed backend must CLAIM the emitted markers -- compile them to
    `sumcheck_round` custom fusions -- not silently decompose them. Every
    byte-gate in this file passes either way (the decomposition is
    byte-identical by the marker contract), so a marker/recognizer drift
    between this checkout and the installed jaxlib would otherwise surface
    only as a many-launch perf cliff at shard scale. GPU-only: the claim is
    the register-resident GPU path's contract (fractalyze/xla#179 merge
    gate); the fusion-config name `sumcheck_round` appears in the optimized
    HLO iff the recognizer claimed the round (the unclaimed decomposition
    inlines the composite away, marker attrs included)."""

    def _assert_claimed(self, fn: Callable[..., Any], *args: Any) -> None:
        if jax.default_backend() != "gpu":
            self.skipTest("claim status is the GPU pairing's contract")
        text = jax.jit(fn).lower(*args).compile().as_text()
        self.assertIn(
            "sumcheck_round", text, "round marker decomposed instead of claiming"
        )

    def test_claims_dense_mid_round(self) -> None:
        planes, eq_int, alpha, scalars, consts, live = _inputs()
        self._assert_claimed(
            lambda p, e, a, s, lv: _composite_fix_and_sum_dense(p, e, a, s, consts, lv),
            planes,
            eq_int,
            alpha,
            scalars,
            live,
        )

    def test_claims_jagged_mid_round(self) -> None:
        pl, er, al, _ga, _ci, _pi, ei, sc, consts, (rc, live, op) = _row_inputs()
        self._assert_claimed(
            lambda pl, er, al, rc, ei, sc, lv: _composite_fix_and_sum_row(
                pl, er, al, rc, ei, sc, consts, lv, op
            ),
            pl,
            er,
            al,
            rc,
            ei,
            sc,
            live,
        )


def _boundary_inputs(
    m: int = 8,
) -> tuple[_Planes, jax.Array, jax.Array, _RoundScalars, _InterpConsts, jax.Array]:
    """A random row->interaction handoff round: the planes enter at width `m`
    (the last row round's padded state) while `eq_int` enters at the post-bind
    width `m // 2` and is NOT folded this round. `live` is the fully-live
    marker (`m // 4` reduce pairs)."""
    planes = _Planes(*(rand_field(s, (m,), KB) for s in range(4)))
    eq_int = rand_field(10, (m // 2,), KB)
    alpha = rand_field(11, (), KB)
    scalars = _RoundScalars(
        eq_adj=rand_field(12, (), KB),
        pad_adj=rand_field(13, (), KB),
        z_cur=rand_field(14, (), KB),
        claim=rand_field(15, (), KB),
        lam=rand_field(16, (), KB),
    )
    consts = _InterpConsts(*_round_interp_constants(KB))
    live = jnp.asarray([m // 4, 0], jnp.int32)
    return planes, eq_int, alpha, scalars, consts, live


class BoundaryRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        planes, eq_int, alpha, scalars, consts, live = _boundary_inputs()
        want = _fix_and_sum_boundary(planes, eq_int, alpha, scalars, consts)
        got = _composite_fix_and_sum_boundary(
            planes, eq_int, alpha, scalars, consts, live
        )
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(
                bool(jnp.all(g == w)), "marked boundary round diverged from eager"
            )

    def test_emits_marker_with_abi(self) -> None:
        planes, eq_int, alpha, scalars, consts, live = _boundary_inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda p, e, a, s, lv: _composite_fix_and_sum_boundary(
                p, e, a, s, consts, lv
            )
        )(planes, eq_int, alpha, scalars, live)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("boundary", text)
        self.assertIn("dense", text)


class FirstRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        pl, ga, ci, pi, er, ei, sc, consts, (rc, live, op) = _round0_inputs()
        self.assertIsNotNone(ga)  # odd segments -> a real re-pad this round
        want = _round_poly_row(pl, ga, ci, pi, er, ei, sc, consts)
        got = _composite_sum_as_poly_row(pl, rc, er, ei, sc, consts, live, op)
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
        pl, ga, ci, pi, er, ei, sc, consts, (rc, live, op) = _round0_inputs(
            counts=(4, 4), nrv=2
        )
        self.assertIsNone(ga)
        want = _round_poly_row(pl, None, ci, pi, er, ei, sc, consts)
        got = _composite_sum_as_poly_row(pl, rc, er, ei, sc, consts, live, op)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(
                bool(jnp.all(g == w)), "no-re-pad first round diverged from eager"
            )

    def test_emits_marker_with_abi(self) -> None:
        pl, _ga, _ci, _pi, er, ei, sc, consts, (rc, live, op) = _round0_inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda pl, rc, er, ei, sc, lv: _composite_sum_as_poly_row(
                pl, rc, er, ei, sc, consts, lv, op
            )
        )(pl, rc, er, ei, sc, live)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("first", text)
        self.assertIn("jagged", text)


class FinalRoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        planes = _Planes(*(rand_field(s, (2,), KB) for s in range(4)))
        alpha = rand_field(11, (), KB)
        want = _fix_last(planes, alpha)
        got = _composite_fix_last(planes, alpha)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertTrue(
                bool(jnp.all(g == w)), "marked final round diverged from eager"
            )

    def test_emits_marker_with_abi(self) -> None:
        planes = _Planes(*(rand_field(s, (2,), KB) for s in range(4)))
        alpha = rand_field(11, (), KB)
        jaxpr = jax.make_jaxpr(_composite_fix_last)(planes, alpha)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("final", text)
        self.assertIn("dense", text)


if __name__ == "__main__":
    absltest.main()
