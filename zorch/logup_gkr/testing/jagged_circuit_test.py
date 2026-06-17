# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import fields

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
    scan_build_jagged_pyramid,
)
from zorch.logup_gkr.testing import build_jagged_pyramid, mixed_field_jagged_layer
from zorch.logup_gkr.testing import random_jagged_layer as _random_jagged_layer
from zorch.testkit.random_field import rand_ext_field, rand_field

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


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


class MixedFieldFirstLayerTest(absltest.TestCase):
    """A first layer may hold base-field numerators under extension-field
    denominators; the transition's `n0*d1 + n1*d0` fold promotes to the common
    field, byte-identically to folding an all-extension copy (zkx#681)."""

    def test_type_accepts_base_numerator_ef_denominator(self) -> None:
        # The shape-only `__post_init__` admits a layer whose numerator and
        # denominator pairs live in different fields.
        layer = mixed_field_jagged_layer(1, (3, 1, 2, 2))
        self.assertEqual(layer.numerator_0.dtype, KB)
        self.assertEqual(layer.denominator_0.dtype, EF)

    def test_transition_promotes_and_matches_all_ef(self) -> None:
        row_counts = (3, 1, 2, 2)
        mixed = mixed_field_jagged_layer(5, row_counts)
        all_ef = JaggedGkrLayer(
            numerator_0=mixed.numerator_0.astype(EF),
            numerator_1=mixed.numerator_1.astype(EF),
            denominator_0=mixed.denominator_0,
            denominator_1=mixed.denominator_1,
            row_counts=row_counts,
        )
        schedule = (2, 1, 1, 1)
        out = jagged_layer_transition(mixed, schedule)
        want = jagged_layer_transition(all_ef, schedule)

        # The fold lifts the base numerators into the extension field...
        self.assertEqual(out.numerator_0.dtype, EF)
        self.assertEqual(out.numerator_1.dtype, EF)
        # ...and the folded layer is byte-identical to the all-extension fold,
        # so a base-field first-layer numerator costs nothing past the first
        # transition.
        for name in ("numerator_0", "numerator_1", "denominator_0", "denominator_1"):
            self.assertTrue(
                bool(jnp.all(getattr(out, name) == getattr(want, name))),
                f"{name} diverged",
            )


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


class ScanBuildJaggedPyramidTest(absltest.TestCase):
    """`scan_build_jagged_pyramid` fuses the eager `jagged_layer_transition`
    chain into one `lax.scan`; every generated layer must be byte-identical to
    the eager `build_jagged_pyramid` (sp1-zorch#55)."""

    def _assert_layers_equal(
        self, got: list[JaggedGkrLayer], want: list[JaggedGkrLayer]
    ) -> None:
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want, strict=True):
            self.assertEqual(g.row_counts, w.row_counts)
            for field in fields(JaggedGkrLayer):
                if field.name == "row_counts":
                    continue
                self.assertTrue(
                    bool(jnp.all(getattr(g, field.name) == getattr(w, field.name))),
                    f"{field.name} diverged for row_counts={w.row_counts}",
                )

    def _assert_matches_eager(self, row_counts: tuple[int, ...]) -> None:
        first = _random_jagged_layer(7, row_counts)
        eager = build_jagged_pyramid(first)
        schedules = [layer.row_counts for layer in eager[1:]]
        self._assert_layers_equal(scan_build_jagged_pyramid(first, schedules), eager)

    def test_matches_eager_small(self) -> None:
        self._assert_matches_eager((3, 1, 5, 2))

    def test_matches_eager_deeper(self) -> None:
        self._assert_matches_eager((7, 3, 5, 2, 6, 1, 4, 8))

    def test_matches_eager_base_field_first_layer(self) -> None:
        # zkx#681: a base-field first-layer numerator under EF denominators. A
        # base scan `init` can't ride `lax.scan` -- step 0's fold promotes it to
        # EF, so carry-out dtype != carry-in. The rolled build must carve the
        # first transition out and stay byte-identical to the eager build, whose
        # first transition does the same base->EF promotion.
        row_counts = (3, 1, 5, 2)
        height = sum(row_counts)
        first = JaggedGkrLayer(
            numerator_0=rand_field(7, (height,), KB),
            numerator_1=rand_field(8, (height,), KB),
            denominator_0=rand_ext_field(9, (height,), KB, EF),
            denominator_1=rand_ext_field(10, (height,), KB, EF),
            row_counts=row_counts,
        )
        eager = build_jagged_pyramid(first)
        scanned = scan_build_jagged_pyramid(
            first, [layer.row_counts for layer in eager[1:]]
        )
        # The first layer keeps base-field numerators; the promoted remainder is EF.
        self.assertEqual(scanned[0].numerator_0.dtype, KB)
        self.assertEqual(scanned[1].numerator_0.dtype, EF)
        self._assert_layers_equal(scanned, eager)


if __name__ == "__main__":
    absltest.main()
