# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    JaggedGkrLayer,
    extract_jagged_outputs,
    jagged_layer_transition,
    layer_transition,
)
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear_mont


def _random_jagged_layer(seed: int, row_counts: tuple[int, ...]) -> JaggedGkrLayer:
    height = sum(row_counts)
    return JaggedGkrLayer(
        numerator_0=rand_field(seed, (height,), KB),
        numerator_1=rand_field(seed + 1, (height,), KB),
        denominator_0=rand_field(seed + 2, (height,), KB),
        denominator_1=rand_field(seed + 3, (height,), KB),
        row_counts=row_counts,
    )


def _segment_fraction_sums(layer: JaggedGkrLayer) -> list[Array]:
    """Per-interaction sum of both children's fractions over the segment's rows."""
    starts = layer.start_indices
    sums = []
    for i in range(layer.num_interactions):
        lo, hi = starts[i], starts[i + 1]
        sums.append(
            jnp.sum(layer.numerator_0[lo:hi] / layer.denominator_0[lo:hi])
            + jnp.sum(layer.numerator_1[lo:hi] / layer.denominator_1[lo:hi])
        )
    return sums


class JaggedGkrLayerTest(absltest.TestCase):
    def test_derives_metadata_from_row_counts(self) -> None:
        layer = _random_jagged_layer(1, (3, 1, 2, 2))
        self.assertEqual(layer.num_interactions, 4)
        self.assertEqual(layer.num_interaction_variables, 2)
        self.assertEqual(layer.height, 8)
        self.assertEqual(layer.start_indices, (0, 3, 4, 6, 8))

    def test_rejects_mle_length_not_matching_row_counts(self) -> None:
        with self.assertRaises(ValueError):
            JaggedGkrLayer(
                numerator_0=jnp.ones((5,), KB),
                numerator_1=jnp.ones((5,), KB),
                denominator_0=jnp.ones((5,), KB),
                denominator_1=jnp.ones((5,), KB),
                row_counts=(3, 1, 2, 2),  # height 8 != 5
            )

    def test_rejects_non_power_of_two_interaction_count(self) -> None:
        with self.assertRaises(ValueError):
            JaggedGkrLayer(
                numerator_0=jnp.ones((4,), KB),
                numerator_1=jnp.ones((4,), KB),
                denominator_0=jnp.ones((4,), KB),
                denominator_1=jnp.ones((4,), KB),
                row_counts=(2, 1, 1),
            )

    def test_rejects_mismatched_mle_widths(self) -> None:
        with self.assertRaises(ValueError):
            JaggedGkrLayer(
                numerator_0=jnp.ones((4,), KB),
                numerator_1=jnp.ones((2,), KB),
                denominator_0=jnp.ones((4,), KB),
                denominator_1=jnp.ones((4,), KB),
                row_counts=(2, 2),
            )


class JaggedTransitionTest(absltest.TestCase):
    def test_uniform_even_segments_match_dense_transition(self) -> None:
        # With uniform even row counts and a no-padding schedule, the flat
        # stride-2 fold is exactly the dense transition on the same arrays.
        row_counts = (4, 4, 4, 4)
        jagged = _random_jagged_layer(10, row_counts)
        dense = GkrLayer(
            numerator_0=jagged.numerator_0,
            numerator_1=jagged.numerator_1,
            denominator_0=jagged.denominator_0,
            denominator_1=jagged.denominator_1,
            num_interaction_variables=2,
        )

        out = jagged_layer_transition(jagged, (2, 2, 2, 2))
        want = layer_transition(dense)

        self.assertEqual(out.row_counts, (2, 2, 2, 2))
        self.assertTrue(bool(jnp.all(out.numerator_0 == want.numerator_0)))
        self.assertTrue(bool(jnp.all(out.numerator_1 == want.numerator_1)))
        self.assertTrue(bool(jnp.all(out.denominator_0 == want.denominator_0)))
        self.assertTrue(bool(jnp.all(out.denominator_1 == want.denominator_1)))

    def test_odd_segment_prepad_preserves_fraction_sums(self) -> None:
        layer = _random_jagged_layer(20, (3, 1, 2, 1))
        out = jagged_layer_transition(layer, (2, 1, 1, 1))
        for got, want in zip(
            _segment_fraction_sums(out), _segment_fraction_sums(layer)
        ):
            self.assertTrue(bool(got == want))

    def test_post_padding_slots_are_fold_neutral(self) -> None:
        layer = _random_jagged_layer(30, (3, 1, 2, 1))
        out = jagged_layer_transition(layer, (4, 2, 2, 2))
        # Padding both preserves the per-segment sums...
        for got, want in zip(
            _segment_fraction_sums(out), _segment_fraction_sums(layer)
        ):
            self.assertTrue(bool(got == want))
        # ...and fills the slots past each folded count with (n=0, d=1).
        folded = (2, 1, 1, 1)
        starts = out.start_indices
        for i, rc in enumerate(folded):
            lo, hi = starts[i] + rc, starts[i + 1]
            self.assertTrue(bool(jnp.all(out.numerator_0[lo:hi] == 0)))
            self.assertTrue(bool(jnp.all(out.numerator_1[lo:hi] == 0)))
            self.assertTrue(bool(jnp.all(out.denominator_0[lo:hi] == 1)))
            self.assertTrue(bool(jnp.all(out.denominator_1[lo:hi] == 1)))

    def test_rejects_schedule_length_mismatch(self) -> None:
        layer = _random_jagged_layer(40, (2, 2, 2, 2))
        with self.assertRaises(ValueError):
            jagged_layer_transition(layer, (1, 1))

    def test_rejects_truncating_schedule(self) -> None:
        # A target below the folded count would silently drop fractions.
        layer = _random_jagged_layer(50, (4, 4, 4, 4))
        with self.assertRaises(ValueError):
            jagged_layer_transition(layer, (1, 2, 2, 2))


class ExtractJaggedOutputsTest(absltest.TestCase):
    def test_all_ones_interleaves_children(self) -> None:
        layer = _random_jagged_layer(60, (1, 1, 1, 1))
        out = extract_jagged_outputs(layer)
        want_num = jnp.stack([layer.numerator_0, layer.numerator_1], -1).flatten()
        want_den = jnp.stack([layer.denominator_0, layer.denominator_1], -1).flatten()
        self.assertTrue(bool(jnp.all(out.numerator == want_num)))
        self.assertTrue(bool(jnp.all(out.denominator == want_den)))

    def test_rejects_uniform_above_floor_row_counts(self) -> None:
        layer = _random_jagged_layer(70, (2, 2, 2, 2))
        with self.assertRaises(ValueError):
            extract_jagged_outputs(layer)

    def test_rejects_mixed_row_counts(self) -> None:
        layer = _random_jagged_layer(80, (2, 1, 1, 1))
        with self.assertRaises(ValueError):
            extract_jagged_outputs(layer)


class JaggedEndToEndTest(absltest.TestCase):
    def test_fraction_sum_invariant_down_to_outputs(self) -> None:
        # Folding to the floor with a minimal consumer schedule preserves the
        # total fraction sum, independent of any padding policy.
        layer = _random_jagged_layer(90, (5, 2, 3, 1))
        total = sum(_segment_fraction_sums(layer))
        while max(layer.row_counts) > 1:
            schedule = tuple((rc + 1) // 2 for rc in layer.row_counts)
            layer = jagged_layer_transition(layer, schedule)
        out = extract_jagged_outputs(layer)
        self.assertTrue(bool(total == jnp.sum(out.numerator / out.denominator)))


if __name__ == "__main__":
    absltest.main()
