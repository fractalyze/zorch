"""composite_or_inline emits one composite marker, or inlines the decomposition."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

import zorch._composite as _composite
from zorch._composite import composite_or_inline


def _composite_eqns(fn: object, *args: jax.Array) -> list:
    """The composite primitive eqns in `fn`'s jaxpr — read without MLIR lowering,
    so this runs on a jaxlib that lacks stablehlo.CompositeOp too."""
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    return [e for e in jaxpr.eqns if e.primitive.name == "composite"]


class CompositeOrInlineTest(absltest.TestCase):
    def test_emits_one_named_composite_carrying_attrs(self) -> None:
        # make_jaxpr captures the marker without lowering, so force the composite
        # path regardless of the running jaxlib.
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = True
            eqns = _composite_eqns(
                lambda x: composite_or_inline(
                    lambda a, **_: a + a, x, name="zorch.t", k=3
                ),
                jnp.arange(4),
            )
        finally:
            _composite._HAS_COMPOSITE_OP = orig
        self.assertLen(eqns, 1)
        self.assertEqual(eqns[0].params["name"], "zorch.t")
        attrs = {key: leaves[0] for key, leaves, _ in eqns[0].params["attributes"]}
        self.assertEqual(attrs["k"], 3)

    def test_inline_path_drops_marker_and_runs_decomposition(self) -> None:
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = False
            x = jnp.arange(4)
            eqns = _composite_eqns(
                lambda v: composite_or_inline(lambda a: a + a, v, name="zorch.t"), x
            )
            out = composite_or_inline(lambda a: a + a, x, name="zorch.t")
        finally:
            _composite._HAS_COMPOSITE_OP = orig
        self.assertEmpty(eqns)
        self.assertTrue(bool(jnp.array_equal(out, x + x)))

    def test_inline_path_passes_attrs_to_decomposition(self) -> None:
        # On the inline path the decomposition still receives the attrs as kwargs
        # — matching how lax.composite calls it when tracing — so a decomposition
        # whose attrs are required keyword arguments works inline too.
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = False
            out = composite_or_inline(
                lambda a, *, scale: a * scale, jnp.arange(4), name="zorch.t", scale=3
            )
        finally:
            _composite._HAS_COMPOSITE_OP = orig
        self.assertTrue(bool(jnp.array_equal(out, jnp.arange(4) * 3)))


if __name__ == "__main__":
    absltest.main()
