"""composite emits one named composite marker carrying its attrs."""

import frx
import frx.numpy as fnp
from absl.testing import absltest

from zorch._composite import composite


def _composite_eqns(fn: object, *args: frx.Array) -> list:
    """The composite primitive eqns in `fn`'s jaxpr — read without MLIR lowering."""
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    return [e for e in jaxpr.eqns if e.primitive.name == "composite"]


class CompositeTest(absltest.TestCase):
    def test_emits_one_named_composite_carrying_attrs(self) -> None:
        eqns = _composite_eqns(
            lambda x: composite(lambda a, **_: a + a, x, name="zorch.t", k=3),
            fnp.arange(4),
        )
        self.assertLen(eqns, 1)
        self.assertEqual(eqns[0].params["name"], "zorch.t")
        attrs = {key: leaves[0] for key, leaves, _ in eqns[0].params["attributes"]}
        self.assertEqual(attrs["k"], 3)


if __name__ == "__main__":
    absltest.main()
