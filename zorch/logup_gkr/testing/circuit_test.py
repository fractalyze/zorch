# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    build_pyramid,
    extract_outputs,
    layer_transition,
)
from zorch.logup_gkr.testing import random_first_layer

KB = zk_dtypes.koalabear


def _rational_sum(numerators: Array, denominators: Array) -> Array:
    """Sum of the fractions numerators[i]/denominators[i], in-field."""
    return jnp.sum(numerators / denominators)


class CircuitTest(absltest.TestCase):
    def test_num_variables_and_row_variables(self) -> None:
        layer = random_first_layer(1, 2, 3)
        self.assertEqual(layer.num_variables, 5)
        self.assertEqual(layer.num_row_variables, 3)

    def test_rejects_mismatched_mle_widths(self) -> None:
        with self.assertRaises(ValueError):
            GkrLayer(
                numerator_0=jnp.ones((4,), KB),
                numerator_1=jnp.ones((2,), KB),
                denominator_0=jnp.ones((4,), KB),
                denominator_1=jnp.ones((4,), KB),
                num_interaction_variables=1,
            )

    def test_rejects_interaction_variables_above_total(self) -> None:
        with self.assertRaises(ValueError):
            GkrLayer(
                numerator_0=jnp.ones((4,), KB),
                numerator_1=jnp.ones((4,), KB),
                denominator_0=jnp.ones((4,), KB),
                denominator_1=jnp.ones((4,), KB),
                num_interaction_variables=3,  # > num_variables (2)
            )

    def test_layer_transition_folds_adjacent_pairs(self) -> None:
        # Hand-computed: with all denominators 1, next_n = n0 + n1 per node,
        # regrouped even -> 0-child, odd -> 1-child.
        ones = jnp.ones((4,), KB)
        layer = GkrLayer(
            numerator_0=jnp.array([1, 2, 3, 4], KB),
            numerator_1=jnp.array([5, 6, 7, 8], KB),
            denominator_0=ones,
            denominator_1=ones,
            num_interaction_variables=1,
        )
        out = layer_transition(layer)
        self.assertEqual(out.num_row_variables, 0)
        self.assertEqual([int(x) for x in out.numerator_0], [6, 10])  # nodes 0,2
        self.assertEqual([int(x) for x in out.numerator_1], [8, 12])  # nodes 1,3
        self.assertEqual([int(x) for x in out.denominator_0], [1, 1])
        self.assertEqual([int(x) for x in out.denominator_1], [1, 1])

    def test_transition_requires_row_variable(self) -> None:
        layer = random_first_layer(20, 2, 0)
        self.assertEqual(layer.num_row_variables, 0)
        with self.assertRaises(ValueError):
            layer_transition(layer)

    def test_build_pyramid_collapses_to_interaction_floor(self) -> None:
        first = random_first_layer(30, 1, 3)
        layers = build_pyramid(first)
        self.assertEqual(len(layers), 4)  # first + one per folded row variable
        self.assertEqual(layers[0].num_row_variables, 3)
        self.assertEqual(layers[-1].num_row_variables, 0)
        self.assertEqual(layers[-1].numerator_0.shape[0], 1 << 1)  # 2^int_vars

    def test_extract_outputs_interleaves_children(self) -> None:
        layer = random_first_layer(40, 2, 0)
        out = extract_outputs(layer)
        want_num = jnp.stack([layer.numerator_0, layer.numerator_1], -1).flatten()
        want_den = jnp.stack([layer.denominator_0, layer.denominator_1], -1).flatten()
        self.assertTrue(bool(jnp.all(out.numerator == want_num)))
        self.assertTrue(bool(jnp.all(out.denominator == want_den)))
        self.assertEqual(out.numerator.shape[0], 1 << 3)  # 2^(int_vars + 1)

    def test_rational_sum_invariant_end_to_end(self) -> None:
        # The GKR fold preserves the sum of all leaf fractions. Deterministic
        # nonzero denominators so every fraction is well defined.
        n0 = jnp.array([3, 7, 11, 2, 5, 9, 13, 4], KB)
        n1 = jnp.array([1, 6, 8, 10, 12, 14, 15, 17], KB)
        d0 = jnp.array([2, 3, 5, 7, 11, 13, 17, 19], KB)
        d1 = jnp.array([23, 29, 31, 37, 41, 43, 47, 53], KB)
        first = GkrLayer(n0, n1, d0, d1, num_interaction_variables=1)

        leaf_sum = _rational_sum(jnp.concatenate([n0, n1]), jnp.concatenate([d0, d1]))
        out = extract_outputs(build_pyramid(first)[-1])
        self.assertTrue(bool(leaf_sum == _rational_sum(out.numerator, out.denominator)))


if __name__ == "__main__":
    absltest.main()
