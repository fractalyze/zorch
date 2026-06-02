"""fused_region runs its decomposition and emits one zorch.round composite."""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.fusion import fused_region
from zorch.testkit.random_field import rand_field


def test_runs_the_decomposition():
    s0 = rand_field(1, (8,), F)
    decomp = lambda s: s + s + s  # straight-line
    assert jnp.array_equal(fused_region(decomp, s0), decomp(s0))


def test_emits_one_zorch_round_composite():
    s0 = rand_field(1, (8,), F)
    decomp = lambda s: s + s
    txt = jax.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
    assert txt.count("stablehlo.composite") == 1, txt
    assert "zorch.round" in txt


if __name__ == "__main__":
    test_runs_the_decomposition()
    test_emits_one_zorch_round_composite()
    print("ok")
