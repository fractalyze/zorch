# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    JaggedGkrLayer,
    _jagged_transition_core,
    build_jagged_pyramid,
    extract_jagged_outputs,
    jagged_layer_transition,
    layer_transition,
)
from zorch.logup_gkr.testing import (
    build_jagged_pyramid as eager_jagged_pyramid,
)
from zorch.logup_gkr.testing import (
    host_counts,
    mixed_field_jagged_layer,
    widen_jagged_layer,
)
from zorch.logup_gkr.testing import random_jagged_layer as _random_jagged_layer
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_ext_field, rand_field

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _segment_fraction_sums(layer: JaggedGkrLayer) -> list[Array]:
    """Per-interaction sum of both children's fractions over the segment's rows."""
    counts = host_counts(layer)
    starts = [0]
    for rc in counts:
        starts.append(starts[-1] + rc)
    sums = []
    for i in range(layer.num_batches):
        lo, hi = starts[i], starts[i + 1]
        sums.append(
            fnp.sum(layer.numerator_0[lo:hi] / layer.denominator_0[lo:hi])
            + fnp.sum(layer.numerator_1[lo:hi] / layer.denominator_1[lo:hi])
        )
    return sums


class JaggedGkrLayerTest(absltest.TestCase):
    def test_derives_metadata_from_shapes(self) -> None:
        layer = _random_jagged_layer(1, (3, 1, 2, 2))
        self.assertEqual(layer.num_batches, 4)
        self.assertEqual(layer.num_batch_variables, 2)
        self.assertEqual(layer.width, 8)
        self.assertEqual(host_counts(layer), (3, 1, 2, 2))

    def test_admits_capacity_slack(self) -> None:
        # width > sum(row_counts) is the capacity contract, not an error.
        layer = widen_jagged_layer(_random_jagged_layer(2, (3, 1, 2, 2)), 12)
        self.assertEqual(layer.width, 12)
        self.assertEqual(host_counts(layer), (3, 1, 2, 2))

    def test_rejects_non_power_of_two_interaction_count(self) -> None:
        with self.assertRaises(ValueError):
            JaggedGkrLayer(
                numerator_0=fnp.ones((4,), KB),
                numerator_1=fnp.ones((4,), KB),
                denominator_0=fnp.ones((4,), KB),
                denominator_1=fnp.ones((4,), KB),
                row_counts=fnp.asarray((2, 1, 1), fnp.int32),
            )

    def test_rejects_mismatched_mle_widths(self) -> None:
        with self.assertRaises(ValueError):
            JaggedGkrLayer(
                numerator_0=fnp.ones((4,), KB),
                numerator_1=fnp.ones((2,), KB),
                denominator_0=fnp.ones((4,), KB),
                denominator_1=fnp.ones((4,), KB),
                row_counts=fnp.asarray((2, 2), fnp.int32),
            )

    def test_aot_lower_admits_arginfo_leaves(self) -> None:
        # `jit(f).lower(layer)` rebuilds the layer with `frx.stages.ArgInfo`
        # leaves (shape/dtype only, no ndim); `__post_init__` must accept
        # them or every compile-only cache warm dies at the lowering step.
        layer = _random_jagged_layer(3, (3, 1, 2, 2))
        frx.jit(lambda la: la.numerator_0).lower(layer)


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
            num_batch_variables=2,
        )

        out = jagged_layer_transition(jagged, (2, 2, 2, 2))
        want = layer_transition(dense)

        self.assertEqual(host_counts(out), (2, 2, 2, 2))
        self.assertTrue(bool(fnp.all(out.numerator_0 == want.numerator_0)))
        self.assertTrue(bool(fnp.all(out.numerator_1 == want.numerator_1)))
        self.assertTrue(bool(fnp.all(out.denominator_0 == want.denominator_0)))
        self.assertTrue(bool(fnp.all(out.denominator_1 == want.denominator_1)))

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
        counts = host_counts(out)
        starts = [0]
        for rc in counts:
            starts.append(starts[-1] + rc)
        for i, rc in enumerate(folded):
            lo, hi = starts[i] + rc, starts[i + 1]
            self.assertTrue(bool(fnp.all(out.numerator_0[lo:hi] == 0)))
            self.assertTrue(bool(fnp.all(out.numerator_1[lo:hi] == 0)))
            self.assertTrue(bool(fnp.all(out.denominator_0[lo:hi] == 1)))
            self.assertTrue(bool(fnp.all(out.denominator_1[lo:hi] == 1)))

    def test_rejects_schedule_length_mismatch(self) -> None:
        layer = _random_jagged_layer(40, (2, 2, 2, 2))
        with self.assertRaises(ValueError):
            jagged_layer_transition(layer, (1, 1))


class JaggedTransitionJitTest(absltest.TestCase):
    """The transition's array core rides a `@jit` island so each pyramid layer
    is one fused dispatch; the fused output must byte-match the eager
    (`frx.disable_jit`) one across the schedule shapes a build hits."""

    def _assert_eager_matches_jit(
        self, layer: JaggedGkrLayer, schedule: tuple[int, ...]
    ) -> None:
        with frx.disable_jit():
            eager = jagged_layer_transition(layer, schedule)
        fused = jagged_layer_transition(layer, schedule)
        self.assertTrue(bool(fnp.all(fused.row_counts == eager.row_counts)))
        for name in ("numerator_0", "numerator_1", "denominator_0", "denominator_1"):
            self.assertTrue(
                bool(fnp.all(getattr(fused, name) == getattr(eager, name))),
                f"{name} diverged",
            )

    def test_uniform_even_no_gather(self) -> None:
        # prepad == row_counts and folded == schedule, so both gathers are the
        # no-op None branch and the island is just the fold.
        self._assert_eager_matches_jit(
            _random_jagged_layer(11, (4, 4, 4, 4)), (2, 2, 2, 2)
        )

    def test_odd_segment_prepad(self) -> None:
        self._assert_eager_matches_jit(
            _random_jagged_layer(12, (3, 1, 2, 1)), (2, 1, 1, 1)
        )

    def test_post_padding(self) -> None:
        self._assert_eager_matches_jit(
            _random_jagged_layer(13, (3, 1, 2, 1)), (4, 2, 2, 2)
        )

    def test_mixed_base_field_numerator_promotes(self) -> None:
        # The base->EF promotion happens inside the island (numerator and
        # denominator avals differ), the path the first build transition takes.
        self._assert_eager_matches_jit(
            mixed_field_jagged_layer(14, (3, 1, 2, 2)), (2, 1, 1, 1)
        )

    def test_saturated_floor(self) -> None:
        # The saturated-floor transition a consumer folds before extracting
        # outputs: every segment at two slots down to one, both gathers no-op.
        self._assert_eager_matches_jit(
            _random_jagged_layer(15, (2, 2, 2, 2)), (1, 1, 1, 1)
        )

    def test_static_counts_warm_reuse_one_trace(self) -> None:
        # The island's load-bearing property: the static count tuples key the
        # compile cache by value, so two independently built layers of one
        # shape warm-reuse a single trace instead of re-tracing per call.
        schedule = (4, 2, 1, 1)
        assert_single_trace(
            self,
            _jagged_transition_core,
            [
                lambda: jagged_layer_transition(
                    _random_jagged_layer(16, (5, 3, 1, 1)), schedule
                ),
                lambda: jagged_layer_transition(
                    _random_jagged_layer(17, (5, 3, 1, 1)), schedule
                ),
            ],
        )


class MixedFieldFirstLayerTest(absltest.TestCase):
    """A first layer may hold base-field numerators under extension-field
    denominators; the transition's `n0*d1 + n1*d0` fold promotes to the common
    field, byte-identically to folding an all-extension copy."""

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
            row_counts=mixed.row_counts,
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
                bool(fnp.all(getattr(out, name) == getattr(want, name))),
                f"{name} diverged",
            )


class ExtractJaggedOutputsTest(absltest.TestCase):
    def test_all_ones_interleaves_children(self) -> None:
        layer = _random_jagged_layer(60, (1, 1, 1, 1))
        out = extract_jagged_outputs(layer)
        want_num = fnp.stack([layer.numerator_0, layer.numerator_1], -1).flatten()
        want_den = fnp.stack([layer.denominator_0, layer.denominator_1], -1).flatten()
        self.assertTrue(bool(fnp.all(out.numerator == want_num)))
        self.assertTrue(bool(fnp.all(out.denominator == want_den)))

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
        while max(host_counts(layer)) > 1:
            schedule = tuple((rc + 1) // 2 for rc in host_counts(layer))
            layer = jagged_layer_transition(layer, schedule)
        out = extract_jagged_outputs(layer)
        self.assertTrue(bool(total == fnp.sum(out.numerator / out.denominator)))


class BuildJaggedPyramidTest(absltest.TestCase):
    """`build_jagged_pyramid` folds the `jagged_layer_transition` chain into one
    unrolled traced region; every generated layer must be byte-identical to the
    eager reference (`eager_jagged_pyramid`)."""

    def _assert_layers_equal(
        self, got: list[JaggedGkrLayer], want: list[JaggedGkrLayer]
    ) -> None:
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want, strict=True):
            self.assertEqual(host_counts(g), host_counts(w))
            for field in fields(JaggedGkrLayer):
                if field.name == "row_counts":
                    continue
                self.assertTrue(
                    bool(fnp.all(getattr(g, field.name) == getattr(w, field.name))),
                    f"{field.name} diverged for row_counts={host_counts(w)}",
                )

    def _assert_matches_eager(self, row_counts: tuple[int, ...]) -> None:
        first = _random_jagged_layer(7, row_counts)
        eager = eager_jagged_pyramid(first)
        schedules = [host_counts(layer) for layer in eager[1:]]
        self._assert_layers_equal(build_jagged_pyramid(first, schedules), eager)

    def test_matches_eager_small(self) -> None:
        self._assert_matches_eager((3, 1, 5, 2))

    def test_matches_eager_deeper(self) -> None:
        self._assert_matches_eager((7, 3, 5, 2, 6, 1, 4, 8))

    def test_matches_eager_base_field_first_layer(self) -> None:
        # A base-field first-layer numerator under EF denominators.
        # Transition 0's fold promotes it to EF; the unrolled build handles the
        # dtype change inline (a `lax.scan` could not -- carry-out dtype would
        # differ from carry-in) and stays byte-identical to the eager build.
        row_counts = (3, 1, 5, 2)
        height = sum(row_counts)
        first = JaggedGkrLayer(
            numerator_0=rand_field(7, (height,), KB),
            numerator_1=rand_field(8, (height,), KB),
            denominator_0=rand_ext_field(9, (height,), KB, EF),
            denominator_1=rand_ext_field(10, (height,), KB, EF),
            row_counts=fnp.asarray(row_counts, fnp.int32),
        )
        eager = eager_jagged_pyramid(first)
        built = build_jagged_pyramid(first, [host_counts(layer) for layer in eager[1:]])
        # The first layer keeps base-field numerators; the promoted remainder is EF.
        self.assertEqual(built[0].numerator_0.dtype, KB)
        self.assertEqual(built[1].numerator_0.dtype, EF)
        self._assert_layers_equal(built, eager)


class CapacityTransitionTest(absltest.TestCase):
    """A transition at slack capacity (traced schedule + explicit out_width)
    must byte-match the zero-slack layout on the live prefix and zero the
    dead region past `sum(out_row_counts)`."""

    def _assert_matches_zero_slack(
        self,
        layer: JaggedGkrLayer,
        schedule: tuple[int, ...],
        in_slack: int = 5,
        out_slack: int = 3,
    ) -> None:
        want = jagged_layer_transition(layer, schedule)
        wide = widen_jagged_layer(layer, layer.width + in_slack)
        out_width = sum(schedule) + out_slack
        got = jagged_layer_transition(wide, fnp.asarray(schedule, fnp.int32), out_width)
        height = sum(schedule)
        self.assertEqual(got.width, out_width)
        for name in (
            "numerator_0",
            "numerator_1",
            "denominator_0",
            "denominator_1",
        ):
            got_a = getattr(got, name)
            want_a = getattr(want, name)
            self.assertEqual(got_a.dtype, want_a.dtype, f"{name} dtype")
            self.assertTrue(
                bool(fnp.all(got_a[:height] == want_a)), f"{name} live prefix"
            )
            self.assertTrue(bool(fnp.all(got_a[height:] == 0)), f"{name} dead tail")

    def test_uniform_even_segments(self) -> None:
        self._assert_matches_zero_slack(
            _random_jagged_layer(210, (4, 4, 4, 4)), (2, 2, 2, 2)
        )

    def test_odd_segment_prepad(self) -> None:
        self._assert_matches_zero_slack(
            _random_jagged_layer(220, (3, 1, 5, 2)), (2, 1, 3, 1)
        )

    def test_post_padding_slots_stay_fold_neutral(self) -> None:
        # Slots between a segment's folded count and its out count are LIVE
        # neutral padding (the next transition reads them); only past
        # sum(out_row_counts) is the dead zero region.
        self._assert_matches_zero_slack(
            _random_jagged_layer(230, (3, 1, 2, 1)), (4, 2, 2, 2)
        )

    def test_mixed_base_field_numerator_promotes(self) -> None:
        self._assert_matches_zero_slack(
            mixed_field_jagged_layer(240, (3, 1, 2, 2)), (2, 1, 1, 1)
        )

    def test_saturated_floor(self) -> None:
        self._assert_matches_zero_slack(
            _random_jagged_layer(250, (2, 2, 2, 2)), (1, 1, 1, 1)
        )

    def test_traced_schedule_requires_out_width(self) -> None:
        layer = _random_jagged_layer(260, (2, 2, 2, 2))
        with self.assertRaises(ValueError):
            jagged_layer_transition(layer, fnp.asarray((1, 1, 1, 1), fnp.int32))

    def test_shares_one_trace_across_layouts(self) -> None:
        # The load-bearing property: two DIFFERENT layouts at one capacity
        # shape reuse a single trace -- the layout values live in traced
        # operands, never in the compile key.
        out_width = 12

        def make_call(
            seed: int, rc: tuple[int, ...], sched: tuple[int, ...]
        ) -> Callable[[], None]:
            def _call() -> None:
                jagged_layer_transition(
                    widen_jagged_layer(_random_jagged_layer(seed, rc), 16),
                    fnp.asarray(sched, fnp.int32),
                    out_width,
                )

            return _call

        assert_single_trace(
            self,
            _jagged_transition_core,
            [
                make_call(280, (5, 3, 1, 1), (4, 2, 2, 2)),
                make_call(281, (2, 6, 1, 3), (2, 4, 2, 2)),
                make_call(282, (1, 1, 1, 9), (2, 2, 2, 6)),
            ],
        )


class CapacityPyramidTest(absltest.TestCase):
    def test_matches_zero_slack_pyramid_and_extract(self) -> None:
        # The eager zero-slack pyramid (the testing policy) rebuilt with the
        # SAME policy evaluated on the traced counts at slack widths: every
        # level's live prefix must byte-match, and the floor must extract
        # identically once its width lands exactly on num_batches.
        row_counts = (5, 3, 1, 7)
        first = _random_jagged_layer(300, row_counts)
        static_layers = eager_jagged_pyramid(first)

        wide_first = widen_jagged_layer(first, first.width + 4)
        schedules = []
        cur = wide_first.row_counts
        for i, level in enumerate(static_layers[1:]):
            # The testing policy on traced counts: fold, then even-pad
            # unsaturated segments. The floor takes zero slack so the
            # capacity extract gate (width == num_batches) opens.
            folded = cur + 1 >> 1
            sched = fnp.where(folded == 1, folded, folded + (folded & 1))
            slack = 0 if i == len(static_layers) - 2 else 2
            schedules.append((sched, sum(host_counts(level)) + slack))
            cur = sched
        got_layers = build_jagged_pyramid(wide_first, schedules)

        self.assertEqual(len(got_layers), len(static_layers))
        for got, want in zip(got_layers[1:], static_layers[1:], strict=True):
            height = sum(host_counts(want))
            self.assertEqual(host_counts(got), host_counts(want))
            for name in (
                "numerator_0",
                "numerator_1",
                "denominator_0",
                "denominator_1",
            ):
                self.assertTrue(
                    bool(fnp.all(getattr(got, name)[:height] == getattr(want, name))),
                    f"{name} diverged at row_counts={host_counts(want)}",
                )

        got_out = extract_jagged_outputs(got_layers[-1])
        want_out = extract_jagged_outputs(static_layers[-1])
        self.assertTrue(bool(fnp.all(got_out.numerator == want_out.numerator)))
        self.assertTrue(bool(fnp.all(got_out.denominator == want_out.denominator)))


if __name__ == "__main__":
    absltest.main()
