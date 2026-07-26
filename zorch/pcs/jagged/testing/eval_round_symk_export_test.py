# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic column-count export of the WHOLE eval round byte-matches concrete.

``eval_round_core`` runs every column-indexed step — the outer indicator's
searchsorted gather, the outer ``Σ D·J̃`` Hadamard sumcheck, and the inner
branching-program sumcheck — over the REAL column count, so ONE ``frx.export``
binary serves every column count at real-size cost (no padding). This locks the
full round: a single symbolic binary must produce a byte-identical
``JaggedEvalMsg`` to the concrete core for two distinct column counts of one area
tier (same ``n_d`` => same round counts, same dense size). The concrete core is
what the SP1-byte-matched ``prover_test`` drives, so equivalence transitively
pins the symbolic path to SP1. Mont-u32, no tolerances.
"""

from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array, export
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.jagged.dense import log_area_tier
from zorch.pcs.jagged.poly import _offset_bit_tensor, build_jagged_layout
from zorch.pcs.jagged.prover import eval_round_core, merged_prefix_bits
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.testkit.transcript import cheap_transcript

_N_R = 5
_LAYOUTS = {6: [20, 20, 20, 20, 15, 5], 8: [13, 13, 13, 13, 12, 12, 12, 12]}
_AREA = 100
_N_D = log_area_tier(_AREA)
_DENSE_LEN = 1 << (_AREA - 1).bit_length()  # 128 >= total area, shared across L
_PRIME = 2013265921


def _u32(x: Array) -> list[int]:
    return np.asarray(frx.lax.bitcast_convert_type(x, fnp.uint32)).reshape(-1).tolist()


def _rand_ef(seed: int, shape: tuple[int, ...]) -> Array:
    ints = np.random.default_rng(seed).integers(
        1, 1 << 30, size=(*shape, 4), dtype=np.int64
    )
    return frx.lax.bitcast_convert_type(fnp.array(ints, dtype=BF), EF)


class SymbolicColumnEvalRoundExportTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.z_col = _rand_ef(1, (3,))  # n_c = 3 bracket
        self.z_row = _rand_ef(2, (_N_R,))
        self.dense = fnp.asarray(
            np.random.default_rng(9).integers(0, _PRIME, (_DENSE_LEN,), np.uint32)
        ).view(BF)

    def _inputs(self, heights: list[int]) -> tuple[Array, Array, Array, Array]:
        """(offsets, merged, weights, all_claims) as prove_jagged_eval builds them."""
        l_max = len(heights)
        _, n_d = build_jagged_layout(heights, l_max, EF)
        assert n_d == _N_D
        offsets = _offset_bit_tensor(heights, l_max, n_d, EF)
        merged = merged_prefix_bits(heights, n_d, dtype=EF)
        weights = expand_eq_to_hypercube(self.z_col, fnp.ones((), EF))[:l_max]
        all_claims = _rand_ef(100 + l_max, (l_max,))
        return offsets, merged, weights, all_claims

    def _fn(
        self,
        offsets: Array,
        merged: Array,
        weights: Array,
        all_claims: Array,
        transcript: Any,
    ) -> Any:
        return eval_round_core(
            offsets,
            merged,
            weights,
            all_claims,
            self.dense,
            self.z_row,
            self.z_col,
            transcript,
            dtype=EF,
        )

    def _export(self) -> export.Exported:
        (length,) = export.symbolic_shape("l", constraints=["l >= 5", "l <= 8"])
        offsets_abs = frx.ShapeDtypeStruct((length + 1, _N_D), EF)
        merged_abs = frx.ShapeDtypeStruct((length, 2 * _N_D), EF)
        weights_abs = frx.ShapeDtypeStruct((length,), EF)
        claims_abs = frx.ShapeDtypeStruct((length,), EF)
        tr_abs = frx.tree_util.tree_map(
            lambda a: frx.ShapeDtypeStruct(a.shape, a.dtype), cheap_transcript(BF)
        )
        return export.export(frx.jit(self._fn))(
            offsets_abs, merged_abs, weights_abs, claims_abs, tr_abs
        )

    def test_one_binary_byte_matches_concrete_for_every_l(self) -> None:
        exported = self._export()
        for length, heights in _LAYOUTS.items():
            offsets, merged, weights, claims = self._inputs(heights)
            self.assertEqual(merged.shape[0], length)

            ref = self._fn(offsets, merged, weights, claims, cheap_transcript(BF))
            got = exported.call(offsets, merged, weights, claims, cheap_transcript(BF))

            ref_leaves = frx.tree_util.tree_leaves(ref)
            got_leaves = frx.tree_util.tree_leaves(got)
            self.assertEqual(len(ref_leaves), len(got_leaves), f"leaf count L={length}")
            for i, (a, b) in enumerate(zip(ref_leaves, got_leaves, strict=True)):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at L={length}")


_ND_HEIGHTS = [3, 5, 2]  # area 10, fits any n_d >= 4
_ND_DENSE = 16  # >= area


class SymbolicNrEvalRoundExportTest(absltest.TestCase):
    """n_r (row-capacity bits = z_row length) symbolic: the row eq is computed per
    element (a lax.scan over n_r bits, no 2^n_r table) and the BP layer count
    ``max(n_r, n_d)`` / ``_bit`` reads flow from the symbolic z_row length. One
    binary byte-matches the concrete round at two row-bit widths (n_r < n_d here,
    so the heights stay representable)."""

    _ND = 6  # fixed n_d; n_r below stays < n_d

    def setUp(self) -> None:
        super().setUp()
        heights = _ND_HEIGHTS  # tallest 5 -> needs n_r >= 3
        self.col = len(heights)
        self.z_col = _rand_ef(1, (2,))
        self.offsets = _offset_bit_tensor(heights, self.col, self._ND, EF)
        self.merged = merged_prefix_bits(heights, self._ND, dtype=EF)
        self.weights = expand_eq_to_hypercube(self.z_col, fnp.ones((), EF))[: self.col]
        self.all_claims = _rand_ef(4, (self.col,))
        self.dense = fnp.asarray(
            np.random.default_rng(9).integers(0, _PRIME, (_ND_DENSE,), np.uint32)
        ).view(BF)

    def _fn(self, z_row: Array, transcript: Any) -> Any:
        return eval_round_core(
            self.offsets,
            self.merged,
            self.weights,
            self.all_claims,
            self.dense,
            z_row,
            self.z_col,
            transcript,
            dtype=EF,
        )

    def _export(self) -> export.Exported:
        (r,) = export.symbolic_shape("r", constraints=["r >= 3", "r <= 5"])
        z_row_abs = frx.ShapeDtypeStruct((r,), EF)
        tr_abs = frx.tree_util.tree_map(
            lambda a: frx.ShapeDtypeStruct(a.shape, a.dtype), cheap_transcript(BF)
        )
        return export.export(frx.jit(self._fn))(z_row_abs, tr_abs)

    def test_one_binary_byte_matches_concrete_for_every_nr(self) -> None:
        exported = self._export()
        for n_r in (3, 5):
            z_row = _rand_ef(20 + n_r, (n_r,))
            ref = self._fn(z_row, cheap_transcript(BF))
            got = exported.call(z_row, cheap_transcript(BF))

            ref_leaves = frx.tree_util.tree_leaves(ref)
            got_leaves = frx.tree_util.tree_leaves(got)
            self.assertEqual(len(ref_leaves), len(got_leaves), f"leaf count n_r={n_r}")
            for i, (a, b) in enumerate(zip(ref_leaves, got_leaves, strict=True)):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at n_r={n_r}")


if __name__ == "__main__":
    absltest.main()
