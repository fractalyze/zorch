"""fused_region runs its decomposition and emits one zorch.fused_region composite.

On a jaxlib without `stablehlo.CompositeOp`, `fused_region` drops the marker and
runs the decomposition inline — so it works under `@jit` everywhere, and the
emission check below is skipped where the op is unavailable.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

import zorch._composite as _composite
from zorch.fusion import fused_region
from zorch.testkit.random_field import rand_field


class FusedRegionTest(absltest.TestCase):
    def test_runs_the_decomposition(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s + s  # straight-line
        self.assertTrue(bool(jnp.array_equal(fused_region(decomp, s0), decomp(s0))))

    def test_runs_under_jit(self) -> None:
        # The path that breaks on a CompositeOp-less wheel: composite *lowering*
        # under @jit. Must produce the decomposition's result on either path.
        s0 = rand_field(2, (8,), F)
        decomp = lambda s: s + s + s
        out = jax.jit(lambda v: fused_region(decomp, v))(s0)
        self.assertTrue(bool(jnp.array_equal(out, decomp(s0))))

    def test_inline_fallback_drops_marker_and_stays_correct(self) -> None:
        # Force the no-CompositeOp path on any jaxlib: no composite is emitted and
        # the result still equals the decomposition.
        s0 = rand_field(3, (8,), F)
        decomp = lambda s: s + s
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = False
            txt = jax.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
            self.assertEqual(txt.count("stablehlo.composite"), 0, txt)
            out = jax.jit(lambda v: fused_region(decomp, v))(s0)
            self.assertTrue(bool(jnp.array_equal(out, decomp(s0))))
        finally:
            _composite._HAS_COMPOSITE_OP = orig

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_emits_one_zorch_fused_region_composite(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s
        txt = jax.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn("zorch.fused_region", txt)


if __name__ == "__main__":
    absltest.main()
