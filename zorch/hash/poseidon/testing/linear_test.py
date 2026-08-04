# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Poseidon's normal-form linear layers: fusion-ready, and each reading its
chained input a bounded number of times.

The rule and the 22-round permutation that produced it are in
`zorch.hash.linear`; the limits below are its per-layer instances.
"""

from functools import partial

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from zorch.hash.poseidon.linear import (
    apply_dense_mds,
    apply_sparse_partial,
    apply_sparse_partial_ints,
)
from zorch.testkit.fusion import (
    assert_fusion_ready,
    assert_input_uses,
    primitive_count,
)
from zorch.testkit.random_field import rand_field

_WIDTHS = (8, 16)


def _int_rows(w: int) -> tuple[tuple[int, ...], ...]:
    """A `w x w` matrix of small canonical ints — the layers only care that the
    coefficients are literals, not what they are."""
    return tuple(tuple((i * w + j) % 97 + 1 for j in range(w)) for i in range(w))


def _sparse_int_coeffs(w: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`(dot_row, col_vec)` as canonical Python ints, for the int-literal twin."""
    return (
        tuple((3 * j + 1) % 97 + 1 for j in range(w)),
        tuple((5 * j + 2) % 97 + 1 for j in range(w - 1)),
    )


def _as_field(values: object) -> fnp.ndarray:
    """Canonical ints (any nesting) -> a field array."""
    return fnp.array(np.array(values, dtype=np.uint64), dtype=F)


def _sparse_args(w: int) -> tuple:
    """`(dot_row, col_vec, active, tail)` for a width-`w` partial round."""
    return (
        rand_field(1, (w,), F),
        rand_field(2, (w - 1,), F),
        rand_field(3, (), F),
        rand_field(4, (w - 1,), F),
    )


class DenseMdsTest(absltest.TestCase):
    def test_equals_the_matrix_form(self) -> None:
        for w in _WIDTHS:
            rows = _int_rows(w)
            s = rand_field(5, (w,), F)
            m = _as_field(rows)
            self.assertTrue(
                bool(fnp.array_equal(apply_dense_mds(rows, s), m @ s)), f"width {w}"
            )

    def test_state_reads_stay_linear_in_width(self) -> None:
        # Indexing `state` inside both loops costs w**2 reads; hoisting the
        # lanes once costs w.
        for w in _WIDTHS:
            rows = _int_rows(w)
            assert_input_uses(
                partial(apply_dense_mds, rows), rand_field(5, (w,), F), limit=w
            )

    def test_is_fusion_ready(self) -> None:
        w = 16
        rows = _int_rows(w)
        assert_fusion_ready(lambda v: apply_dense_mds(rows, v), rand_field(5, (w,), F))


class SparsePartialTest(absltest.TestCase):
    def test_equals_the_documented_normal_form(self) -> None:
        for w in _WIDTHS:
            dot_row, col_vec, active, tail = _sparse_args(w)
            got = apply_sparse_partial(dot_row, col_vec, active, tail)
            # Independent in form, not a transcription of the unrolled fold: a
            # test reference carries no fusion constraint, so it may reduce.
            want0 = dot_row[0] * active + fnp.sum(dot_row[1:] * tail)
            want = fnp.concatenate([want0[None], tail + col_vec * active])
            self.assertTrue(bool(fnp.array_equal(got, want)), f"width {w}")

    def test_reads_the_shared_lane_value_twice(self) -> None:
        # `active` is the post-S-box lane-0 value every lane consumes: once in
        # the lane-0 dot, once in the rank-1 update. A per-lane form reads it
        # w-1 times, and each read re-derives the whole preceding round.
        for w in _WIDTHS:
            assert_input_uses(apply_sparse_partial, *_sparse_args(w), arg=2, limit=2)

    def test_multiplies_do_not_scale_with_width(self) -> None:
        # The same rule seen through the op count: 2w-1 multiplies per-lane
        # versus a constant few in array form. Sharper than the read count here,
        # and it fails naming the width that drifted.
        counts = {
            w: primitive_count(apply_sparse_partial, *_sparse_args(w), name="mul")
            for w in (8, 12, 16)
        }
        self.assertEqual(
            len(set(counts.values())),
            1,
            msg=(
                f"multiplies must not scale with width, got {counts}. A per-lane "
                f"scalar form gives 2w-1; see apply_sparse_partial."
            ),
        )

    def test_is_fusion_ready(self) -> None:
        assert_fusion_ready(apply_sparse_partial, *_sparse_args(16))


class SparsePartialIntsTest(absltest.TestCase):
    def test_matches_its_field_array_twin(self) -> None:
        for w in _WIDTHS:
            ints_dot, ints_col = _sparse_int_coeffs(w)
            _, _, active, tail = _sparse_args(w)
            self.assertTrue(
                bool(
                    fnp.array_equal(
                        apply_sparse_partial_ints(ints_dot, ints_col, active, tail),
                        apply_sparse_partial(
                            _as_field(ints_dot), _as_field(ints_col), active, tail
                        ),
                    )
                ),
                f"width {w}",
            )

    def test_reads_the_shared_lane_value_once_per_lane(self) -> None:
        # The documented exception: this twin knowingly does NOT hold the rule.
        # Broadcasting `active` first would hold it without capturing a field
        # array; it is left per-lane because that costs ~23% more traced
        # equations and nothing chains this body at a width where it matters
        # (see apply_sparse_partial_ints). `w` is the current count, not slack,
        # so this fails if it gets worse and tightens for free if it is fixed.
        for w in _WIDTHS:
            ints_dot, ints_col = _sparse_int_coeffs(w)
            _, _, active, tail = _sparse_args(w)
            assert_input_uses(
                partial(apply_sparse_partial_ints, ints_dot, ints_col),
                active,
                tail,
                limit=w,
            )


if __name__ == "__main__":
    absltest.main()
