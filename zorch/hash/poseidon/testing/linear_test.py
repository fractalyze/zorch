# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Poseidon's normal-form linear layers: fusion-ready, and holding the
chained-input rule in `zorch.hash.linear`.

The rule is about *scaling*, so every check runs at two widths: a layer that
re-reads the value the rounds thread through makes the compiler re-derive it per
read, and where rounds chain that compounds per round instead of adding. That is
what stopped a 22-round Goldilocks permutation from finishing
(https://github.com/fractalyze/zorch/issues/565).
"""

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from zorch.hash.poseidon.linear import (
    apply_dense_mds,
    apply_sparse_partial,
    apply_sparse_partial_ints,
)
from zorch.testkit.fusion import assert_fusion_ready, assert_input_uses
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


def _as_field(values: tuple[int, ...]) -> fnp.ndarray:
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
            m = fnp.array([[c for c in row] for row in rows], dtype=F)
            self.assertTrue(
                bool(fnp.array_equal(apply_dense_mds(rows, s), m @ s)), f"width {w}"
            )

    def test_state_reads_stay_linear_in_width(self) -> None:
        # Indexing `state` inside both loops costs w**2 reads; hoisting the lanes
        # once costs w. Two widths, because it is the scaling that matters.
        for w in _WIDTHS:
            rows = _int_rows(w)
            assert_input_uses(
                lambda v, rows=rows: apply_dense_mds(rows, v),
                rand_field(5, (w,), F),
                limit=w,
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
            want0 = dot_row[0] * active
            for j in range(1, w):
                want0 = want0 + dot_row[j] * tail[j - 1]
            want = fnp.concatenate([want0[None], tail + col_vec * active])
            self.assertTrue(bool(fnp.array_equal(got, want)), f"width {w}")

    def test_reads_the_shared_lane_value_a_fixed_number_of_times(self) -> None:
        # `active` is the post-S-box lane-0 value every lane consumes. Per-lane
        # scalar expressions read it w-1 times and each read re-derives the whole
        # preceding round; the array form reads it a fixed few.
        for w in _WIDTHS:
            dot_row, col_vec, active, tail = _sparse_args(w)
            assert_input_uses(
                lambda a, d=dot_row, c=col_vec, t=tail: apply_sparse_partial(
                    d, c, a, t
                ),
                active,
                limit=4,
            )

    def test_multiplies_do_not_scale_with_width(self) -> None:
        # The same rule seen through the op count: 2w-1 multiplies per-lane
        # versus a constant few in array form. Sharper than the read count here,
        # and it fails naming the width that drifted.
        counts = {}
        for w in (8, 12, 16):
            dot_row, col_vec, active, tail = _sparse_args(w)
            jaxpr = frx.make_jaxpr(apply_sparse_partial)(
                dot_row, col_vec, active, tail
            ).jaxpr
            counts[w] = sum(1 for eqn in jaxpr.eqns if eqn.primitive.name == "mul")
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

    def test_shared_lane_value_reads_stay_within_width(self) -> None:
        # The known exception: the int-literal twin still reads `active` once per
        # lane, because its operand ABI forbids the field array that makes the
        # array form possible (broadcasting the scalar would hold the rule at ~23%
        # more traced equations). Bounded at `w` rather than pinned, so giving it
        # the broadcast later tightens this without editing the test.
        for w in _WIDTHS:
            ints_dot, ints_col = _sparse_int_coeffs(w)
            _, _, active, tail = _sparse_args(w)
            assert_input_uses(
                lambda a, d=ints_dot, c=ints_col, t=tail: apply_sparse_partial_ints(
                    d, c, a, t
                ),
                active,
                limit=w,
            )


if __name__ == "__main__":
    absltest.main()
