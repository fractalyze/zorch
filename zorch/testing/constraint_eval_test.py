"""constraint_eval runs eval + α-RLC and emits one zorch.constraint_eval composite."""

import frx
import frx.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.constraint_eval import CONSTRAINT_EVAL_MARKER, constraint_eval
from zorch.testkit.random_field import rand_field


def _eval_fn(rows: frx.Array) -> frx.Array:
    """A straight-line stand-in for a per-row constraint evaluation:
    rows [N, num_cols] -> constraints [N, K]. Self-contained (no scheme/zkVM
    knowledge) so the test anchors only on its own golden."""
    c0 = rows[:, 0] * rows[:, 1]
    c1 = rows[:, 1] + rows[:, 2]
    c2 = rows[:, 0] * rows[:, 2] + rows[:, 1]
    return jnp.stack([c0, c1, c2], axis=-1)


def _eval_fn_aux(rows: frx.Array, aux: frx.Array) -> frx.Array:
    """A 2-ary constraint stand-in that reads an auxiliary operand beyond the
    trace: the aux-threading counterpart to `_eval_fn`. Each leg mixes an aux
    element so a wrong value (or a dropped operand) breaks the golden."""
    c0 = rows[:, 0] * rows[:, 1] + aux[0]
    c1 = rows[:, 1] + rows[:, 2]
    c2 = rows[:, 0] * rows[:, 2] + rows[:, 1] * aux[1]
    return jnp.stack([c0, c1, c2], axis=-1)


class ConstraintEvalTest(absltest.TestCase):
    def test_folds_to_the_same_rlc_as_a_plain_dot(self) -> None:
        # The composite must inline to the identical result as the plain
        # `eval_fn(rows) @ alpha` it replaces — exact (field add is
        # associative), so the fold's association order is irrelevant.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        golden = _eval_fn(rows) @ alpha
        got = constraint_eval(_eval_fn, rows, alpha)
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    def test_empty_alpha_raises(self) -> None:
        rows = rand_field(1, (8, 3), F)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn, rows, rand_field(2, (0,), F))

    def test_emits_one_zorch_constraint_eval_composite(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            frx.jit(lambda t, a: constraint_eval(_eval_fn, t, a))
            .lower(rows, alpha)
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(CONSTRAINT_EVAL_MARKER, txt)
        # K is carried as a composite attribute for the XLA-side recognizer.
        self.assertIn("num_constraints", txt)
        # Without a live-width bound the marker must NOT declare one — the
        # declaration routes XLA to the bounded emitter.
        self.assertNotIn("live_width_operand_idx", txt)

    def test_live_width_masks_rows_past_the_bound(self) -> None:
        # Rows at index >= live_width read as the field's zero — the same
        # masked form the bounded emitter's else branch produces, so marked
        # and inlined paths stay byte-identical lane for lane.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        golden = _eval_fn(rows) @ alpha
        golden = jnp.where(jnp.arange(8) < 5, golden, jnp.zeros_like(golden))
        got = constraint_eval(_eval_fn, rows, alpha, live_width=5)
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    def test_live_width_at_full_height_keeps_every_row(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        unbounded = constraint_eval(_eval_fn, rows, alpha)
        bounded = constraint_eval(_eval_fn, rows, alpha, live_width=8)
        self.assertTrue(bool(jnp.array_equal(bounded, unbounded)))

    def test_live_width_zero_masks_every_row(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        got = constraint_eval(_eval_fn, rows, alpha, live_width=0)
        self.assertTrue(bool(jnp.array_equal(got, jnp.zeros_like(got))))

    def test_live_width_accepts_a_traced_scalar(self) -> None:
        # The bound is a runtime value by design (per-round values share one
        # kernel), so it must trace as an argument, not bake in as a constant.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        golden = constraint_eval(_eval_fn, rows, alpha, live_width=5)
        got = frx.jit(lambda t, a, lw: constraint_eval(_eval_fn, t, a, live_width=lw))(
            rows, alpha, jnp.int32(5)
        )
        self.assertTrue(bool(jnp.array_equal(got, golden)))

    def test_live_width_rejects_bad_bounds(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn, rows, alpha, live_width=-1)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn, rows, alpha, live_width=jnp.array([5], jnp.int32))
        with self.assertRaises(ValueError):
            # A field scalar is not the s32 wire type XLA validates.
            constraint_eval(_eval_fn, rows, alpha, live_width=rand_field(3, (), F))
        with self.assertRaises(ValueError):
            # live_width validation rejects a non-int32 (float) before the frx
            # asarray funnel: "live_width must be a scalar int32".
            constraint_eval(_eval_fn, rows, alpha, live_width=1.5)
        with self.assertRaises(ValueError):
            # A scalar result has no leading row axis to bound.
            constraint_eval(
                lambda t: jnp.stack([t[0] * t[1]]),
                rand_field(4, (3,), F),
                rand_field(2, (1,), F),
                live_width=1,
            )

    def test_live_width_attr_rides_the_composite(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            frx.jit(lambda t, a, lw: constraint_eval(_eval_fn, t, a, live_width=lw))
            .lower(rows, alpha, jnp.int32(5))
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(CONSTRAINT_EVAL_MARKER, txt)
        # The bound rides as operand 2; XLA routes on this declaration and
        # hard-errors unless that operand is a scalar s32.
        self.assertIn("live_width_operand_idx = 2", txt)

    def test_column_weights_adds_the_weighted_column_sum(self) -> None:
        # column_weights folds `sum_c trace[:, c] * w[c]` into each row's value,
        # under the live mask: dead rows zero out of the column term too. A
        # window into a compact-packed shared buffer straddles the NEXT chip's
        # live rows, so dead rows are NOT zero and an unmasked dot would leak
        # them — the live-bounded emitter kernel zeroes whole dead rows, and
        # the inlined decomposition must match it byte-for-byte. Random rows
        # (dead rows are NOT zero here) pin that the mask covers the dot.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        weights = rand_field(5, (3,), F)  # one weight per trace column
        fold = _eval_fn(rows) @ alpha + rows @ weights
        golden = jnp.where(jnp.arange(8) < 5, fold, jnp.zeros_like(fold))
        got = constraint_eval(
            _eval_fn, rows, alpha, live_width=5, column_weights=weights
        )
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    @absltest.skip(
        "jit elides the field transpose in the column-weight dot, so jit output "
        "byte-differs from eager on the published frx nightly. The XLA-pass "
        "fix is in XLA, not the released wheel."
    )
    def test_column_weights_under_jit_matches_eager(self) -> None:
        # The column term is a dot inside the marked body; confirm the jitted /
        # lowered path equals the eager golden (the emitter folds the dot, the
        # inlined path runs it directly — neither may diverge).
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        weights = rand_field(5, (3,), F)
        golden = constraint_eval(
            _eval_fn, rows, alpha, live_width=5, column_weights=weights
        )
        got = frx.jit(
            lambda t, a, w: constraint_eval(
                _eval_fn, t, a, live_width=5, column_weights=w
            )
        )(rows, alpha, weights)
        self.assertTrue(bool(jnp.array_equal(got, golden)))

    def test_column_weights_requires_live_width(self) -> None:
        # column_weights rides as the trailing operand after live_width, so the
        # bounded path is mandatory (keeps the operand order fixed).
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(
                _eval_fn, rows, alpha, column_weights=rand_field(5, (3,), F)
            )

    def test_column_weights_rejects_wrong_shape(self) -> None:
        # One weight per trace column (rank-1, len == trace.shape[-1]); a mismatch
        # must fail loud at the entry point, not as a cryptic matmul trace error.
        rows = rand_field(1, (8, 3), F)  # 3 columns
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):  # wrong length (2 != 3)
            constraint_eval(
                _eval_fn,
                rows,
                alpha,
                live_width=5,
                column_weights=rand_field(5, (2,), F),
            )
        with self.assertRaises(ValueError):  # wrong rank (2D, not 1D)
            constraint_eval(
                _eval_fn,
                rows,
                alpha,
                live_width=5,
                column_weights=rand_field(6, (3, 1), F),
            )

    def test_aux_threads_through_to_a_two_ary_eval_fn(self) -> None:
        # With one aux operand, eval_fn is called as eval_fn(trace, aux); the
        # composite must inline to the identical `eval_fn(rows, aux) @ alpha`.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        aux = rand_field(7, (2,), F)
        golden = _eval_fn_aux(rows, aux) @ alpha
        got = constraint_eval(_eval_fn_aux, rows, alpha, aux_operands=(aux,))
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    def test_aux_declared_operand_survives_jit_where_a_closure_breaks(self) -> None:
        # Why aux is an operand and not a closure: under frx.jit an array closed
        # into eval_fn reaches the decomposition as a Tracer constant, which
        # lax.composite rejects. Declared, the same computation traces cleanly.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        aux = rand_field(7, (2,), F)
        with self.assertRaises(frx.errors.UnexpectedTracerError):
            frx.jit(
                lambda t, a, x: constraint_eval(lambda tr: _eval_fn_aux(tr, x), t, a)
            )(rows, alpha, aux)
        golden = _eval_fn_aux(rows, aux) @ alpha
        got = frx.jit(
            lambda t, a, x: constraint_eval(_eval_fn_aux, t, a, aux_operands=(x,))
        )(rows, alpha, aux)
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    def test_aux_operand_idxs_attr_rides_the_composite(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        aux = rand_field(7, (2,), F)
        txt = (
            frx.jit(
                lambda t, a, x: constraint_eval(_eval_fn_aux, t, a, aux_operands=(x,))
            )
            .lower(rows, alpha, aux)
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(CONSTRAINT_EVAL_MARKER, txt)
        # No live_width/column_weights, so the lone aux is at operand 2.
        self.assertIn("aux_operand_idxs = [2]", txt)

    def test_aux_absent_declares_no_operand_idxs(self) -> None:
        # No aux operands ⇒ no declaration; that attr routes XLA to the
        # aux-threading emitter path.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            frx.jit(lambda t, a: constraint_eval(_eval_fn, t, a))
            .lower(rows, alpha)
            .as_text()
        )
        self.assertNotIn("aux_operand_idxs", txt)

    def test_aux_rejects_a_bare_array(self) -> None:
        # A single array (not a sequence) would splat into per-element scalars.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(
                _eval_fn_aux, rows, alpha, aux_operands=rand_field(7, (2,), F)
            )

    def test_aux_rejects_none(self) -> None:
        # A `pv=None`-style migration slip must fail loud, not as a len(None)
        # crash deeper in the function.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn_aux, rows, alpha, aux_operands=None)  # type: ignore[arg-type]

    def test_aux_accepts_a_list(self) -> None:
        # A caller naturally passing a list (not a tuple) is normalized, not
        # met with a cryptic tuple-concatenation error.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        aux = rand_field(7, (2,), F)
        golden = _eval_fn_aux(rows, aux) @ alpha
        got = constraint_eval(_eval_fn_aux, rows, alpha, aux_operands=[aux])  # type: ignore[arg-type]
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))

    def test_multiple_aux_operands_thread_in_order(self) -> None:
        # Two aux operands feed eval_fn(trace, a0, a1) and ride at consecutive
        # trailing indices.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        a0 = rand_field(7, (2,), F)
        a1 = rand_field(8, (2,), F)

        def eval2(r: frx.Array, x0: frx.Array, x1: frx.Array) -> frx.Array:
            return _eval_fn_aux(r, x0) + _eval_fn_aux(r, x1)

        golden = eval2(rows, a0, a1) @ alpha
        got = constraint_eval(eval2, rows, alpha, aux_operands=(a0, a1))
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))
        txt = (
            frx.jit(
                lambda t, a, x0, x1: constraint_eval(eval2, t, a, aux_operands=(x0, x1))
            )
            .lower(rows, alpha, a0, a1)
            .as_text()
        )
        self.assertIn("aux_operand_idxs = [2, 3]", txt)

    def test_aux_composes_with_live_width(self) -> None:
        # aux rides after live_width, so its index shifts to 3; the masked
        # golden must still match lane for lane.
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        aux = rand_field(7, (2,), F)
        golden = _eval_fn_aux(rows, aux) @ alpha
        golden = jnp.where(jnp.arange(8) < 5, golden, jnp.zeros_like(golden))
        got = constraint_eval(
            _eval_fn_aux, rows, alpha, live_width=5, aux_operands=(aux,)
        )
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))
        txt = (
            frx.jit(
                lambda t, a, lw, x: constraint_eval(
                    _eval_fn_aux, t, a, live_width=lw, aux_operands=(x,)
                )
            )
            .lower(rows, alpha, jnp.int32(5), aux)
            .as_text()
        )
        self.assertIn("live_width_operand_idx = 2", txt)
        self.assertIn("aux_operand_idxs = [3]", txt)

    def test_aux_composes_with_live_width_and_column_weights(self) -> None:
        # All optionals present: order is (trace, alpha, live, weights, aux), so
        # aux's index shifts to 4. The eager result equals the live mask over
        # the fold PLUS the column term (mask-last: dead rows zero out of the
        # dot too, matching the live-bounded emitter kernel).
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        weights = rand_field(5, (3,), F)
        aux = rand_field(7, (2,), F)
        fold = _eval_fn_aux(rows, aux) @ alpha + rows @ weights
        golden = jnp.where(jnp.arange(8) < 5, fold, jnp.zeros_like(fold))
        got = constraint_eval(
            _eval_fn_aux,
            rows,
            alpha,
            live_width=5,
            column_weights=weights,
            aux_operands=(aux,),
        )
        self.assertTrue(bool(jnp.array_equal(got, golden)), (got, golden))
        txt = (
            frx.jit(
                lambda t, a, lw, w, x: constraint_eval(
                    _eval_fn_aux,
                    t,
                    a,
                    live_width=lw,
                    column_weights=w,
                    aux_operands=(x,),
                )
            )
            .lower(rows, alpha, jnp.int32(5), weights, aux)
            .as_text()
        )
        self.assertIn("live_width_operand_idx = 2", txt)
        self.assertIn("aux_operand_idxs = [4]", txt)

    def test_start_offset_windows_a_taller_buffer(self) -> None:
        # A small constrained trace of height h, embedded at row `off` in a
        # taller (T-row) shared buffer whose other rows are garbage (both the
        # dead window tail and the rows outside the window entirely); windowing
        # at (off, window_rows=W) with live_width=h must byte-equal evaluating
        # the zero-padded window directly — the window slice, not the
        # surrounding garbage, is what the constraint sees. Cover the boundary
        # offsets 0 and T-W (the first/last window that fits) plus an interior
        # one, so dynamic_slice edge/clamp behavior is pinned.
        h, W, nc, T = 5, 8, 3, 20
        small = rand_field(9, (h, nc), F)
        alpha = rand_field(2, (3,), F)
        # zero-pad rows [h:W) (Montgomery zero is all-zero bytes, a valid
        # field zero), so the window is the live rows followed by a zero tail.
        window = jnp.concatenate([small, jnp.zeros((W - h, nc), F)], axis=0)
        want = constraint_eval(
            _eval_fn, window, alpha, live_width=jnp.asarray(h, jnp.int32)
        )
        for off in (0, 6, T - W):
            with self.subTest(off=off):
                head = rand_field(10 + off, (off, nc), F)  # rows before the window
                # small at [off, off+h); the rest of the buffer is garbage — the
                # in-window tail [off+h, off+W) is masked by live_width, the rest
                # is outside the window.
                tail = rand_field(30 + off, (T - off - h, nc), F)
                tall = jnp.concatenate([head, small, tail], axis=0)
                got = constraint_eval(
                    _eval_fn,
                    tall,
                    alpha,
                    live_width=jnp.asarray(h, jnp.int32),
                    start_offset=jnp.asarray(off, jnp.int32),
                    window_rows=W,
                )
                self.assertTrue(bool(jnp.array_equal(got, want)), (off, got, want))

    def test_col_stride_windows_a_jagged_flat_buffer(self) -> None:
        # A [h, nc] chip trace packed COLUMN-MAJOR into a flat 1-D buffer:
        # column c's rows at flat[off + c*H + r], every other element garbage
        # (the in-column dead tail r in [h, H) and everything outside). The
        # jagged window (start_offset=off, col_stride=H, num_cols=nc,
        # window_rows=W) with live_width=h + column_weights must byte-equal the
        # zero-padded rank-2 window — pins the flat gather, the mask-last dead
        # rows, and the column dot on the jagged path.
        h, W, H, off, nc = 5, 6, 9, 7, 3
        small = rand_field(9, (h, nc), F)
        alpha = rand_field(2, (3,), F)
        weights = rand_field(5, (nc,), F)
        parts = [rand_field(40, (off,), F)]
        for c in range(nc):
            parts += [small[:, c], rand_field(50 + c, (H - h,), F)]
        parts.append(rand_field(60, (4,), F))
        flat = jnp.concatenate(parts)
        window = jnp.concatenate([small, jnp.zeros((W - h, nc), F)], axis=0)
        want = constraint_eval(
            _eval_fn,
            window,
            alpha,
            live_width=jnp.asarray(h, jnp.int32),
            column_weights=weights,
        )
        got = constraint_eval(
            _eval_fn,
            flat,
            alpha,
            live_width=jnp.asarray(h, jnp.int32),
            start_offset=jnp.asarray(off, jnp.int32),
            window_rows=W,
            col_stride=jnp.asarray(H, jnp.int32),
            num_cols=nc,
            column_weights=weights,
        )
        self.assertTrue(bool(jnp.array_equal(got, want)), (got, want))

    def test_fold_operands_fail_loud_on_wrong_type_or_field(self) -> None:
        # The decomposition and the emitter both evaluate base + k*delta in
        # the trace's field; a float k, a k of another field, or a delta of
        # another field must fail at the contract boundary, not corrupt
        # downstream.
        h, W = 5, 8
        tall = rand_field(3, (W + 2, 2), F)
        delta = rand_field(4, (W + 2, 2), F)
        alpha = rand_field(5, (3,), F)

        def call(**kw: object) -> frx.Array:
            merged = dict(
                live_width=h,
                start_offset=0,
                window_rows=W,
                delta=delta,
                fold_coeff=jnp.zeros((), F),
            )
            merged.update(kw)
            return constraint_eval(_eval_fn, tall, alpha, **merged)

        with self.assertRaises(TypeError):
            call(fold_coeff=1.5)
        with self.assertRaises(ValueError):
            call(fold_coeff=jnp.zeros((), jnp.int32))
        with self.assertRaises(ValueError):
            call(delta=delta.view(jnp.uint32))

    def test_col_stride_composes_with_the_fold(self) -> None:
        # Jagged base + delta flat buffers, fold coefficient k: the marker's
        # in-window fold `base + k*delta` must byte-equal evaluating the
        # pre-folded zero-padded rank-2 window directly.
        h, W, H, off, nc = 4, 4, 6, 3, 3
        base2 = rand_field(11, (h, nc), F)
        delta2 = rand_field(12, (h, nc), F)
        alpha = rand_field(2, (3,), F)
        k = rand_field(13, (), F)

        def pack(mat: frx.Array, seed: int) -> frx.Array:
            parts = [rand_field(seed, (off,), F)]
            for c in range(nc):
                parts += [mat[:, c], rand_field(seed + 1 + c, (H - h,), F)]
            return jnp.concatenate(parts)

        flat_b, flat_d = pack(base2, 70), pack(delta2, 80)
        eff = base2 + k * delta2
        window = jnp.concatenate([eff, jnp.zeros((W - h, nc), F)], axis=0)
        want = constraint_eval(
            _eval_fn, window, alpha, live_width=jnp.asarray(h, jnp.int32)
        )
        got = constraint_eval(
            _eval_fn,
            flat_b,
            alpha,
            live_width=jnp.asarray(h, jnp.int32),
            start_offset=jnp.asarray(off, jnp.int32),
            window_rows=W,
            col_stride=jnp.asarray(H, jnp.int32),
            num_cols=nc,
            delta=flat_d,
            fold_coeff=k,
        )
        self.assertTrue(bool(jnp.array_equal(got, want)), (got, want))

    def test_start_offset_attrs_ride_the_composite(self) -> None:
        h, off, W, nc = 5, 3, 8, 3
        tall = rand_field(9, (off + W + 2, nc), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            frx.jit(
                lambda t, a, lw, so: constraint_eval(
                    _eval_fn,
                    t,
                    a,
                    live_width=lw,
                    start_offset=so,
                    window_rows=W,
                )
            )
            .lower(tall, alpha, jnp.int32(h), jnp.int32(off))
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        # start_offset rides right after live_width (operand 2), so its index
        # is 3; window_rows is a static attribute, not an operand.
        self.assertIn("live_width_operand_idx = 2", txt)
        self.assertIn("start_offset_operand_idx = 3", txt)
        self.assertIn(f"window_rows = {W}", txt)

    def test_start_offset_drives_the_dynamic_slice_start_directly(self) -> None:
        # The recognizing emitter binds the window base structurally: the
        # parameter feeding a dynamic-slice's axis-0 start (its
        # ConstraintEvalStartOffsetIdx). JAX's default negative-index wrap would
        # interpose a compare/add/select on that start, hiding the parameter and
        # dropping the marker to the unbounded path. Pin that the decomposition's
        # windowing slice reads the start_offset OPERAND directly — start token is
        # a function %arg, not an SSA temp.
        h, off, W, nc = 5, 3, 8, 3
        tall = rand_field(9, (off + W + 2, nc), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            frx.jit(
                lambda t, a, lw, so: constraint_eval(
                    _eval_fn, t, a, live_width=lw, start_offset=so, window_rows=W
                )
            )
            .lower(tall, alpha, jnp.int32(h), jnp.int32(off))
            .as_text()
        )
        self.assertRegex(txt, r"stablehlo\.dynamic_slice %arg\d+, %arg\d+,", txt)

    def test_start_offset_requires_live_width(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn, rows, alpha, start_offset=0, window_rows=8)

    def test_start_offset_requires_window_rows(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            constraint_eval(_eval_fn, rows, alpha, live_width=8, start_offset=0)

    def test_start_offset_rejects_bad_bounds(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        # Valid live_width + window_rows companions on every call, so it is the
        # bad start_offset — not a missing companion — that each rejects
        # (parity with test_live_width_rejects_bad_bounds).
        with self.assertRaises(ValueError):
            constraint_eval(
                _eval_fn, rows, alpha, live_width=8, start_offset=-1, window_rows=8
            )
        with self.assertRaises(ValueError):
            constraint_eval(
                _eval_fn,
                rows,
                alpha,
                live_width=8,
                start_offset=jnp.array([5], jnp.int32),
                window_rows=8,
            )
        with self.assertRaises(ValueError):
            # A field scalar is not the s32 wire type XLA validates.
            constraint_eval(
                _eval_fn,
                rows,
                alpha,
                live_width=8,
                start_offset=rand_field(3, (), F),
                window_rows=8,
            )
        with self.assertRaises(ValueError):
            # A float is not int32; rejected before the frx asarray funnel.
            constraint_eval(
                _eval_fn, rows, alpha, live_width=8, start_offset=1.5, window_rows=8
            )

    def test_window_rows_requires_start_offset(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            # window_rows alone sizes a window that is never sliced — a no-op.
            constraint_eval(_eval_fn, rows, alpha, live_width=8, window_rows=8)

    def test_window_rows_rejects_bad_values(self) -> None:
        rows = rand_field(1, (8, 3), F)  # trace height 8
        alpha = rand_field(2, (3,), F)
        with self.assertRaises(ValueError):
            # A float is not the Python int the static slice size / attr needs.
            constraint_eval(
                _eval_fn,
                rows,
                alpha,
                live_width=8,
                start_offset=0,
                window_rows=8.5,  # type: ignore[arg-type]
            )
        # bool (an int subclass), zero, negative, and past the trace height.
        for bad in (True, 0, -1, 9):
            with self.subTest(window_rows=bad):
                with self.assertRaises(ValueError):
                    constraint_eval(
                        _eval_fn,
                        rows,
                        alpha,
                        live_width=8,
                        start_offset=0,
                        window_rows=bad,
                    )

    def test_start_offset_operand_order_with_all_optionals(self) -> None:
        # All optionals present — order is (trace, alpha, live, offset, weights,
        # aux) — so start_offset lands at 3 and aux shifts to 5. Pins the
        # dynamic operand-index computation against a reordering regression.
        h, off, W, nc = 5, 3, 8, 3
        tall = rand_field(9, (off + W + 2, nc), F)
        alpha = rand_field(2, (3,), F)
        weights = rand_field(5, (nc,), F)
        aux = rand_field(7, (2,), F)
        txt = (
            frx.jit(
                lambda t, a, lw, so, w, x: constraint_eval(
                    _eval_fn_aux,
                    t,
                    a,
                    live_width=lw,
                    start_offset=so,
                    window_rows=W,
                    column_weights=w,
                    aux_operands=(x,),
                )
            )
            .lower(tall, alpha, jnp.int32(h), jnp.int32(off), weights, aux)
            .as_text()
        )
        self.assertIn("live_width_operand_idx = 2", txt)
        self.assertIn("start_offset_operand_idx = 3", txt)
        self.assertIn(f"window_rows = {W}", txt)
        self.assertIn("aux_operand_idxs = [5]", txt)


if __name__ == "__main__":
    absltest.main()
