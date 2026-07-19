"""fused_region runs its decomposition and emits one zorch.fused_region composite."""

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.fusion import fused_region
from zorch.testkit.random_field import rand_field


class FusedRegionTest(absltest.TestCase):
    def test_runs_the_decomposition(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s + s  # straight-line
        self.assertTrue(bool(fnp.array_equal(fused_region(decomp, s0), decomp(s0))))

    def test_runs_under_jit(self) -> None:
        # composite *lowering* under @jit must produce the decomposition's result.
        s0 = rand_field(2, (8,), F)
        decomp = lambda s: s + s + s
        out = frx.jit(lambda v: fused_region(decomp, v))(s0)
        self.assertTrue(bool(fnp.array_equal(out, decomp(s0))))

    def test_emits_one_zorch_fused_region_composite(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s
        txt = frx.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn("zorch.fused_region", txt)


if __name__ == "__main__":
    absltest.main()
