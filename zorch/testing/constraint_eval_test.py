"""constraint_eval runs eval + α-RLC and emits one zorch.constraint_eval composite."""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch._composite import _HAS_COMPOSITE_OP
from zorch.constraint_eval import CONSTRAINT_EVAL_MARKER, constraint_eval
from zorch.testkit.random_field import rand_field


def _eval_fn(rows: jax.Array) -> jax.Array:
    """A straight-line stand-in for a per-row constraint evaluation:
    rows [N, num_cols] -> constraints [N, K]. Self-contained (no scheme/zkVM
    knowledge) so the test anchors only on its own golden."""
    c0 = rows[:, 0] * rows[:, 1]
    c1 = rows[:, 1] + rows[:, 2]
    c2 = rows[:, 0] * rows[:, 2] + rows[:, 1]
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

    @absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
    def test_emits_one_zorch_constraint_eval_composite(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            jax.jit(lambda t, a: constraint_eval(_eval_fn, t, a))
            .lower(rows, alpha)
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(CONSTRAINT_EVAL_MARKER, txt)
        # K is carried as a composite attribute for the zkx-side recognizer.
        self.assertIn("num_constraints", txt)
        # Without a live-width bound the marker must NOT declare one — the
        # declaration routes zkx to the bounded emitter.
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
        got = jax.jit(lambda t, a, lw: constraint_eval(_eval_fn, t, a, live_width=lw))(
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
            # A field scalar is not the s32 wire type zkx validates.
            constraint_eval(_eval_fn, rows, alpha, live_width=rand_field(3, (), F))
        with self.assertRaises(TypeError):
            # The float-less jax fork rejects the asarray funnel itself.
            constraint_eval(_eval_fn, rows, alpha, live_width=1.5)
        with self.assertRaises(ValueError):
            # A scalar result has no leading row axis to bound.
            constraint_eval(
                lambda t: jnp.stack([t[0] * t[1]]),
                rand_field(4, (3,), F),
                rand_field(2, (1,), F),
                live_width=1,
            )

    @absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
    def test_live_width_attr_rides_the_composite(self) -> None:
        rows = rand_field(1, (8, 3), F)
        alpha = rand_field(2, (3,), F)
        txt = (
            jax.jit(lambda t, a, lw: constraint_eval(_eval_fn, t, a, live_width=lw))
            .lower(rows, alpha, jnp.int32(5))
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(CONSTRAINT_EVAL_MARKER, txt)
        # The bound rides as operand 2; zkx routes on this declaration and
        # hard-errors unless that operand is a scalar s32.
        self.assertIn("live_width_operand_idx = 2", txt)


if __name__ == "__main__":
    absltest.main()
