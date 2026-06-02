# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.circuit import (
    GkrLayer,
    build_pyramid,
    extract_outputs,
    layer_transition,
)
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear
_P = 2130706433  # KoalaBear prime = 2^31 - 2^24 + 1.


# --- pure-Python reference (canonical-int arithmetic, independent of JAX) ---


def _ints(arr):
    return [int(x) for x in arr]


def _ref_fold(n0, n1, d0, d1):
    """Stride-2 fractional-sum fold: even nodes -> new 0-child, odd -> 1-child."""
    rn0, rn1, rd0, rd1 = [], [], [], []
    for i in range(0, len(n0), 2):
        rn0.append((n0[i] * d1[i] + n1[i] * d0[i]) % _P)
        rd0.append((d0[i] * d1[i]) % _P)
        rn1.append((n0[i + 1] * d1[i + 1] + n1[i + 1] * d0[i + 1]) % _P)
        rd1.append((d0[i + 1] * d1[i + 1]) % _P)
    return rn0, rn1, rd0, rd1


def _frac_sum(nums, dens):
    """Sum of nums[i]/dens[i] as one (N, D) fraction mod _P."""
    N, D = 0, 1
    for n, d in zip(nums, dens):
        N = (N * d + n * D) % _P
        D = (D * d) % _P
    return N, D


def _frac_eq(a, b):
    """Cross-multiply equality of two (N, D) fractions mod _P."""
    return (a[0] * b[1] - b[0] * a[1]) % _P == 0


def _layer(seed, num_int_vars, num_row_vars):
    """Random dense first layer with 2^(int+row) flat width per MLE."""
    width = 1 << (num_int_vars + num_row_vars)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), KB),
        numerator_1=rand_field(seed + 1, (width,), KB),
        denominator_0=rand_field(seed + 2, (width,), KB),
        denominator_1=rand_field(seed + 3, (width,), KB),
        num_interaction_variables=num_int_vars,
    )


class CircuitTest(absltest.TestCase):
    def test_num_variables_and_row_variables(self):
        layer = _layer(1, num_int_vars=2, num_row_vars=3)
        self.assertEqual(layer.num_variables, 5)
        self.assertEqual(layer.num_row_variables, 3)

    def test_layer_transition_matches_reference(self):
        layer = _layer(10, num_int_vars=1, num_row_vars=2)
        out = layer_transition(layer)
        ref = _ref_fold(
            _ints(layer.numerator_0),
            _ints(layer.numerator_1),
            _ints(layer.denominator_0),
            _ints(layer.denominator_1),
        )
        self.assertEqual(out.num_row_variables, 1)
        self.assertEqual(_ints(out.numerator_0), ref[0])
        self.assertEqual(_ints(out.numerator_1), ref[1])
        self.assertEqual(_ints(out.denominator_0), ref[2])
        self.assertEqual(_ints(out.denominator_1), ref[3])

    def test_transition_requires_row_variable(self):
        # num_row_variables == 0: width 2^int_vars, nothing left to fold.
        layer = _layer(20, num_int_vars=2, num_row_vars=0)
        self.assertEqual(layer.num_row_variables, 0)
        with self.assertRaises(ValueError):
            layer_transition(layer)

    def test_build_pyramid_collapses_to_interaction_floor(self):
        first = _layer(30, num_int_vars=1, num_row_vars=3)
        layers = build_pyramid(first)
        # first + one per folded row variable.
        self.assertEqual(len(layers), 4)
        self.assertEqual(layers[0].num_row_variables, 3)
        self.assertEqual(layers[-1].num_row_variables, 0)
        self.assertEqual(layers[-1].numerator_0.shape[0], 1 << 1)  # == 2^int_vars

    def test_extract_outputs_interleaves_children(self):
        # row_vars == 0: numerator interleaves (n0, n1), denominator (d0, d1).
        layer = _layer(40, num_int_vars=2, num_row_vars=0)
        out = extract_outputs(layer)
        n0, n1 = _ints(layer.numerator_0), _ints(layer.numerator_1)
        d0, d1 = _ints(layer.denominator_0), _ints(layer.denominator_1)
        want_num = [v for pair in zip(n0, n1) for v in pair]
        want_den = [v for pair in zip(d0, d1) for v in pair]
        self.assertEqual(_ints(out.numerator), want_num)
        self.assertEqual(_ints(out.denominator), want_den)
        self.assertEqual(out.numerator.shape[0], 1 << 3)  # 2^(int_vars + 1)

    def test_rational_sum_invariant_end_to_end(self):
        # Deterministic nonzero denominators so the fraction reference is defined.
        n0 = jnp.array([3, 7, 11, 2, 5, 9, 13, 4], dtype=KB)
        n1 = jnp.array([1, 6, 8, 10, 12, 14, 15, 17], dtype=KB)
        d0 = jnp.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=KB)
        d1 = jnp.array([23, 29, 31, 37, 41, 43, 47, 53], dtype=KB)
        first = GkrLayer(n0, n1, d0, d1, num_interaction_variables=1)

        leaf_nums = _ints(n0) + _ints(n1)
        leaf_dens = _ints(d0) + _ints(d1)
        leaf_sum = _frac_sum(leaf_nums, leaf_dens)

        out = extract_outputs(build_pyramid(first)[-1])
        out_sum = _frac_sum(_ints(out.numerator), _ints(out.denominator))

        self.assertTrue(_frac_eq(leaf_sum, out_sum))


if __name__ == "__main__":
    absltest.main()
