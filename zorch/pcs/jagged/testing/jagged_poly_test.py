# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator
from functools import partial, reduce
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    NUM_BIT_STATES,
    NUM_MEMORY_STATES,
    JaggedStaticConfig,
    _offset_bit_tensor,
    bp_eval_core,
    build_jagged_layout,
    build_prefix_sums,
    eval_jagged_mle,
    partial_eval,
)
from zorch.pcs.jagged.testing import scatter_partial_eval
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.testkit.random_field import rand_field

EF = zk_dtypes.koalabearx4_mont


def _field_val(x: Array) -> bytes:
    """Field scalar → bytes for equality/inequality assertions.

    int() does not work for extension field (koalabearx4), so we use
    numpy bytes comparison instead. Semantically identical to int() for
    base-field scalars.
    """
    return np.array(x).tobytes()


class TransitionTableTest(absltest.TestCase):
    def test_shape_and_onehot(self) -> None:
        # 64 = 4 memory × 16 bit states; each row is one-hot (valid) or zeros (FAIL).
        self.assertEqual(len(_TRANSITION_ROWS), NUM_MEMORY_STATES * NUM_BIT_STATES)
        for row in _TRANSITION_ROWS:
            self.assertEqual(len(row), NUM_MEMORY_STATES)
            self.assertIn(sum(row), (0, 1))


def _bits_msb(val: int, nbits: int, dtype: Any) -> Array:
    return jnp.array([(val >> (nbits - 1 - k)) & 1 for k in range(nbits)], dtype=dtype)


class BpEvalCoreTest(absltest.TestCase):
    def test_single_column_boolean_corner(self) -> None:
        # One column, range [t_c, t_{c+1}) = [0, 4), n_d=3, n_r=2.
        # At boolean points h=1 iff (i = r AND i<4). i=2,r=2 -> 1; i=2,r=1 -> 0.
        dtype = EF
        n_d, n_r = 3, 2
        t_mat = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
        num_vars = max(n_r, n_d)
        pl = _bits_msb(0, n_d, dtype)  # t_c = 0
        pr = _bits_msb(4, n_d, dtype)  # t_{c+1} = 4
        EF1 = jnp.array(1, dtype=dtype)
        EF0 = jnp.array(0, dtype=dtype)
        # match: i=2 (z_index), r=2 (z_row)
        ok = bp_eval_core(
            _bits_msb(2, n_r, dtype), _bits_msb(2, n_d, dtype), pl, pr, t_mat, num_vars
        )
        self.assertEqual(_field_val(ok), _field_val(EF1))
        # mismatch: i=2, r=1 → i - t_c = 2 ≠ r
        no = bp_eval_core(
            _bits_msb(1, n_r, dtype), _bits_msb(2, n_d, dtype), pl, pr, t_mat, num_vars
        )
        self.assertEqual(_field_val(no), _field_val(EF0))
        # out of range: i=5 ≥ t_{c+1}=4 → 0
        oor = bp_eval_core(
            _bits_msb(5, n_r, dtype), _bits_msb(5, n_d, dtype), pl, pr, t_mat, num_vars
        )
        self.assertEqual(_field_val(oor), _field_val(EF0))


class LayoutBuilderTest(absltest.TestCase):
    def test_shape_and_empty_range_pad(self) -> None:
        # heights [4,2,3] → prefix [0,4,6,9], t_L=9, n_d=log2_ceil(9)+1=5.
        cps, cfg = build_jagged_layout([4, 2, 3], l_max=4, n_r=3, dtype=EF)
        self.assertEqual(cps.shape, (5, 5))  # (l_max+1, n_d)
        self.assertEqual((cfg.l_max, cfg.n_c, cfg.n_d), (4, 2, 5))
        # Padding column (c=3): row 3 == row 4 == bits(9) -> empty range.
        self.assertEqual(cps[3].tolist(), cps[4].tolist())

    def test_capacity_assert(self) -> None:
        with self.assertRaises(ValueError):
            build_jagged_layout([4, 2, 3], l_max=2, n_r=3, dtype=EF)  # real_L=3 > 2


def _eq_at(z: Array, idx: int, nbits: int) -> Array:
    # eval_eq(MSB-bits(idx), z) = Π_j (1 - z_j - b_j + 2 z_j b_j)
    bits = [(idx >> (nbits - 1 - j)) & 1 for j in range(nbits)]
    factor = jnp.ones([], dtype=z.dtype)
    for j in range(nbits):
        b = jnp.array(bits[j], dtype=z.dtype)
        factor = factor * (1 - z[j] - b + 2 * z[j] * b)
    return factor


def _naive_jagged_mle(
    row_counts: list[int],
    cfg: JaggedStaticConfig,
    z_row: Array,
    z_col: Array,
    z_index: Array,
) -> Array:
    prefix = build_prefix_sums(row_counts)
    total = jnp.zeros([], dtype=cfg.dtype)
    for c in range(len(row_counts)):
        a_c = prefix[c + 1] - prefix[c]
        for r in range(a_c):
            i = prefix[c] + r
            total = total + (
                _eq_at(z_row, r, cfg.n_r)
                * _eq_at(z_col, c, cfg.n_c)
                * _eq_at(z_index, i, cfg.n_d)
            )
    return total


class EvalJaggedMleTest(absltest.TestCase):
    def test_matches_oracle_compile_once_many_heights(self) -> None:
        # Several height vectors sharing the same (l_max, log-area tier).
        # total_area in [8,16) -> n_d = log2_ceil(area)+1 = 5, same tier.
        heights_list = [[4, 2, 3], [1, 5, 3], [2, 2, 5]]  # areas 9,9,9
        l_max, n_r = 4, 3
        z_row = rand_field(1, (n_r,), EF)
        # cfg is identical across instances (same tier) — z_col/z_index widths too.
        _, cfg = build_jagged_layout(heights_list[0], l_max, n_r, EF)
        z_col = rand_field(2, (cfg.n_c,), EF)
        z_index = rand_field(3, (cfg.n_d,), EF)
        fn = jax.jit(partial(eval_jagged_mle, cfg=cfg))
        for hc in heights_list:
            cps, cfg2 = build_jagged_layout(hc, l_max, n_r, EF)
            self.assertEqual((cfg2.n_d, cfg2.n_c), (cfg.n_d, cfg.n_c))  # same tier
            got = fn(cps, z_row, z_col, z_index)
            want = _naive_jagged_mle(hc, cfg, z_row, z_col, z_index)
            self.assertEqual(_field_val(got), _field_val(want))
        # Compiled once: a single cache entry (no re-trace).
        self.assertEqual(fn._cache_size(), 1)

    def test_zero_bit_pad_diverges(self) -> None:
        # Negative test: zero-bit padding (vs empty-range) changes J̃.
        l_max, n_r = 4, 3
        cps, cfg = build_jagged_layout([4, 2, 3], l_max, n_r, EF)
        z_row = rand_field(1, (n_r,), EF)
        z_col = rand_field(2, (cfg.n_c,), EF)
        z_index = rand_field(3, (cfg.n_d,), EF)
        good = eval_jagged_mle(cps, z_row, z_col, z_index, cfg=cfg)
        bad_cps = cps.at[3].set(jnp.zeros(cfg.n_d, dtype=EF))  # padding row -> t=0
        bad = eval_jagged_mle(bad_cps, z_row, z_col, z_index, cfg=cfg)
        self.assertNotEqual(_field_val(good), _field_val(bad))


class PartialEvalTest(absltest.TestCase):
    def test_partial_eval_matches_eval_jagged_mle(self) -> None:
        heights = [4, 2, 3]
        l_max, n_r = 4, 3
        cps, cfg = build_jagged_layout(heights, l_max, n_r, EF)
        # partial_eval reads its scatter offsets from limb 0 of the tensor it is
        # given; a Montgomery tensor's limb 0 is R mod p, not the canonical bit, so
        # pass the canonical-limb offsets sibling (the production caller's path).
        offsets = _offset_bit_tensor(heights, l_max, cfg)
        z_row = rand_field(10, (cfg.n_r,), EF)
        z_col = rand_field(11, (cfg.n_c,), EF)
        z_index = rand_field(12, (cfg.n_d,), EF)
        indicator = partial_eval(offsets, z_row, z_col, cfg=cfg)
        self.assertEqual(indicator.shape, (2**cfg.n_d,))
        eq_idx = expand_eq_to_hypercube(z_index, jnp.array(1, EF))
        # jnp.sum aborts on EF (koalabearx4_mont) — unroll at trace time.
        n = 2**cfg.n_d
        got = reduce(operator.add, [indicator[i] * eq_idx[i] for i in range(n)])
        # The field-valued mont tensor stays the oracle's input.
        want = eval_jagged_mle(cps, z_row, z_col, z_index, cfg=cfg)
        self.assertEqual(_field_val(got), _field_val(want))

    def test_partial_eval_bit_equals_scatter_oracle(self) -> None:
        # Gather form vs the per-column scatter oracle, byte-for-byte, across
        # the layout corners: zero-height mid column (duplicate prefix entry),
        # single column taller than 2^n_r (capacity truncation), full l_max
        # (tail clamps into a real column), and the standard mixed case.
        l_max, n_r = 4, 3
        layouts = [[4, 2, 3], [4, 0, 3, 2], [9], [3, 3, 2, 1]]
        for heights in layouts:
            with self.subTest(heights=heights):
                _cps, cfg = build_jagged_layout(heights, l_max, n_r, EF)
                offsets = _offset_bit_tensor(heights, l_max, cfg)
                z_row = rand_field(30, (cfg.n_r,), EF)
                z_col = rand_field(31, (cfg.n_c,), EF)
                got = partial_eval(offsets, z_row, z_col, cfg=cfg)
                want = scatter_partial_eval(offsets, z_row, z_col, cfg=cfg)
                self.assertEqual(_field_val(got), _field_val(want))

    def test_partial_eval_all_padding(self) -> None:
        # All columns empty: every prefix entry duplicates t_L = 0, no element
        # is owned, the indicator is identically zero. The all-padding tier is
        # n_d = 1, so any n_r with 2^n_r > 2 makes the scatter form untraceable
        # (its fixed window exceeds the dense buffer) — the gather form has no
        # window, so it also covers that region of the domain.
        l_max = 4
        zeros = [0, 0, 0]
        # n_r=1: window fits the buffer — differential vs the oracle.
        _cps, cfg = build_jagged_layout(zeros, l_max, n_r=1, dtype=EF)
        offsets = _offset_bit_tensor(zeros, l_max, cfg)
        z_row = rand_field(40, (cfg.n_r,), EF)
        z_col = rand_field(41, (cfg.n_c,), EF)
        got = partial_eval(offsets, z_row, z_col, cfg=cfg)
        want = scatter_partial_eval(offsets, z_row, z_col, cfg=cfg)
        self.assertEqual(_field_val(got), _field_val(want))
        self.assertEqual(_field_val(got), _field_val(jnp.zeros(1 << cfg.n_d, dtype=EF)))
        # n_r=3: scatter-untraceable region — pin the gather's zero indicator.
        _cps, cfg = build_jagged_layout(zeros, l_max, n_r=3, dtype=EF)
        offsets = _offset_bit_tensor(zeros, l_max, cfg)
        z_row = rand_field(42, (cfg.n_r,), EF)
        z_col = rand_field(43, (cfg.n_c,), EF)
        got = partial_eval(offsets, z_row, z_col, cfg=cfg)
        self.assertEqual(_field_val(got), _field_val(jnp.zeros(1 << cfg.n_d, dtype=EF)))

    def test_partial_eval_jits_and_compiles_once_across_heights(self) -> None:
        # Two height vectors in the same tier produce the same compiled artifact.
        # The offsets tensor is a traced argument — no ConcretizationError,
        # cache_size==1.
        l_max, n_r = 4, 3
        heights_a = [4, 2, 3]  # area 9, n_d=5
        heights_b = [1, 5, 3]  # area 9, n_d=5 (same tier)
        _cps_a, cfg = build_jagged_layout(heights_a, l_max, n_r, EF)
        _cps_b, cfg_b = build_jagged_layout(heights_b, l_max, n_r, EF)
        self.assertEqual((cfg.n_d, cfg.n_c), (cfg_b.n_d, cfg_b.n_c))  # same tier
        off_a = _offset_bit_tensor(heights_a, l_max, cfg)
        off_b = _offset_bit_tensor(heights_b, l_max, cfg)
        z_row = rand_field(20, (cfg.n_r,), EF)
        z_col = rand_field(21, (cfg.n_c,), EF)
        run = jax.jit(lambda cps, zr, zc: partial_eval(cps, zr, zc, cfg=cfg))
        out1 = run(off_a, z_row, z_col)
        out1.block_until_ready()
        out2 = run(off_b, z_row, z_col)
        out2.block_until_ready()
        # Compiled once: a single cache entry (no re-trace across height vectors).
        self.assertEqual(run._cache_size(), 1)
        # Sanity: the two height vectors produce different indicators.
        self.assertNotEqual(out1.tobytes(), out2.tobytes())


if __name__ == "__main__":
    absltest.main()
