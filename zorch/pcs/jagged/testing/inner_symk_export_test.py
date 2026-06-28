# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic column-count export of the inner BP sumcheck byte-matches concrete.

``inner_sumcheck_core`` does its per-column work as a ``vmap`` + ``jnp.sum`` over
``merged``'s REAL columns, so ONE ``jax.export`` binary serves every column count
``L`` at real-size cost (no padding) — the proper polymorphic form of the column
axis, mirroring ``stacked_basefold_open``'s symbolic ``K``. This locks it: a
single symbolic binary must produce byte-identical sumcheck output to the
concrete core for two distinct ``L`` of the same area tier (so same ``n_d`` /
round count). The concrete core is what the SP1-byte-matched ``prover_test``
drives, so equivalence transitively pins the symbolic path. Mont-u32, no
tolerances.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import Array, export
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.jagged.dense import log_area_tier
from zorch.pcs.jagged.poly import build_jagged_layout
from zorch.pcs.jagged.prover import inner_sumcheck_core, merged_prefix_bits
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.testkit.transcript import cheap_transcript

_N_R = 5  # row-capacity bits (covers the tallest column below)
# Two layouts of one area tier (area 100 -> n_d 8) but different column counts in
# the n_c=3 bracket (L in (4, 8]); the symbolic binary is constrained to it.
_LAYOUTS = {
    6: [20, 20, 20, 20, 15, 5],
    8: [13, 13, 13, 13, 12, 12, 12, 12],
}
_AREA = 100
_N_D = log_area_tier(_AREA)


def _u32(x: Array) -> list[int]:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32)).reshape(-1).tolist()


def _rand_ef(seed: int, shape: tuple[int, ...]) -> Array:
    ints = np.random.default_rng(seed).integers(
        1, 1 << 30, size=(*shape, 4), dtype=np.int64
    )
    return jax.lax.bitcast_convert_type(jnp.array(ints, dtype=BF), EF)


def _build(heights: list[int], z_col: Array) -> tuple[Array, Array]:
    """(merged, weights) for a concrete layout — the same path eval_round_core uses."""
    _, n_d = build_jagged_layout(heights, len(heights), EF)
    assert n_d == _N_D
    merged = merged_prefix_bits(heights, n_d, dtype=EF)
    weights = expand_eq_to_hypercube(z_col, jnp.ones((), EF))[: len(heights)]
    return merged, weights


class SymbolicColumnInnerExportTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.z_col = _rand_ef(1, (3,))  # n_c = 3
        self.z_row = _rand_ef(2, (_N_R,))
        self.z_trace = _rand_ef(3, (2 * _N_D,))

    def _fn(self, merged: Array, weights: Array, transcript: Any) -> Any:
        return inner_sumcheck_core(
            merged,
            weights,
            self.z_row,
            self.z_trace,
            transcript,
            dtype=EF,
            num_bits=_N_D,
        )

    def _export(self) -> export.Exported:
        (length,) = export.symbolic_shape("l", constraints=["l >= 5", "l <= 8"])
        merged_abs = jax.ShapeDtypeStruct((length, 2 * _N_D), EF)
        weights_abs = jax.ShapeDtypeStruct((length,), EF)
        tr_abs = jax.tree_util.tree_map(
            lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), cheap_transcript(BF)
        )
        return export.export(jax.jit(self._fn))(merged_abs, weights_abs, tr_abs)

    def test_one_binary_byte_matches_concrete_for_every_l(self) -> None:
        exported = self._export()
        for length, heights in _LAYOUTS.items():
            merged, weights = _build(heights, self.z_col)
            self.assertEqual(merged.shape[0], length)

            ref = self._fn(merged, weights, cheap_transcript(BF))
            got = exported.call(merged, weights, cheap_transcript(BF))

            ref_leaves = jax.tree_util.tree_leaves(ref)
            got_leaves = jax.tree_util.tree_leaves(got)
            self.assertEqual(len(ref_leaves), len(got_leaves), f"leaf count L={length}")
            for i, (a, b) in enumerate(zip(ref_leaves, got_leaves, strict=True)):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at L={length}")


_T = 8  # z_trace length, >= max n_d in the bracket below


class SymbolicNdInnerExportTest(absltest.TestCase):
    """num_bits (= n_d) symbolic: the inner sumcheck's round count is ``2·n_d`` (a
    lax.scan trip count) and the merged-buffer width ``2·n_d`` — so one binary
    serves every area tier. ``merged_prefix_bits(heights, n_d)`` packs the same
    prefix sums at any ``n_d >= ceil(log2 area)``; this exports over a symbolic
    ``n_d`` and byte-matches the concrete core at two tiers."""

    def setUp(self) -> None:
        super().setUp()
        self.heights = [3, 5, 2]  # area 10, fits any n_d >= 4
        self.col = len(self.heights)
        z_col = _rand_ef(1, (2,))  # n_c = 2 for L = 3
        self.z_row = _rand_ef(2, (_N_R,))  # n_r = 5 < every n_d below
        self.weights = expand_eq_to_hypercube(z_col, jnp.ones((), EF))[: self.col]
        self.z_trace = _rand_ef(3, (_T,))

    def _fn(self, merged: Array, z_trace: Array, transcript: Any) -> Any:
        return inner_sumcheck_core(
            merged,
            self.weights,
            self.z_row,
            z_trace,
            transcript,
            dtype=EF,
            num_bits=merged.shape[1] // 2,
        )

    def _export(self) -> export.Exported:
        (d,) = export.symbolic_shape("d", constraints=["d >= 6", "d <= 8"])
        merged_abs = jax.ShapeDtypeStruct((self.col, 2 * d), EF)
        ztrace_abs = jax.ShapeDtypeStruct((_T,), EF)
        tr_abs = jax.tree_util.tree_map(
            lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), cheap_transcript(BF)
        )
        return export.export(jax.jit(self._fn))(merged_abs, ztrace_abs, tr_abs)

    def test_one_binary_byte_matches_concrete_for_every_nd(self) -> None:
        exported = self._export()
        for n_d in (6, 8):
            merged = merged_prefix_bits(self.heights, n_d, dtype=EF)
            self.assertEqual(merged.shape[1], 2 * n_d)

            ref = self._fn(merged, self.z_trace, cheap_transcript(BF))
            got = exported.call(merged, self.z_trace, cheap_transcript(BF))

            ref_leaves = jax.tree_util.tree_leaves(ref)
            got_leaves = jax.tree_util.tree_leaves(got)
            self.assertEqual(len(ref_leaves), len(got_leaves), f"leaf count n_d={n_d}")
            for i, (a, b) in enumerate(zip(ref_leaves, got_leaves, strict=True)):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at n_d={n_d}")


if __name__ == "__main__":
    absltest.main()
