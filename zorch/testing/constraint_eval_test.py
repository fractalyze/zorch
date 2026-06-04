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


if __name__ == "__main__":
    absltest.main()
