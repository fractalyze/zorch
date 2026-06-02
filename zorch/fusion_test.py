"""fused_region runs its decomposition and emits one zorch.round composite."""

import jax
import jax.numpy as jnp
import pytest
from jaxlib.mlir.dialects import stablehlo
from zk_dtypes import koalabear_mont as F

from zorch.fusion import fused_region
from zorch.testkit.random_field import rand_field

# Lowering a composite needs stablehlo.CompositeOp in jaxlib's MLIR bindings.
# The published jaxlib wheel doesn't carry the fork's backport yet (only a
# self-built jaxlib does), so this emission check is skipped there; eager use
# still works because lax.composite runs its decomposition.
_HAS_COMPOSITE_OP = hasattr(stablehlo, "CompositeOp")


def test_runs_the_decomposition() -> None:
    s0 = rand_field(1, (8,), F)
    decomp = lambda s: s + s + s  # straight-line
    assert jnp.array_equal(fused_region(decomp, s0), decomp(s0))


@pytest.mark.skipif(not _HAS_COMPOSITE_OP, reason="jaxlib lacks stablehlo.CompositeOp")
def test_emits_one_zorch_round_composite() -> None:
    s0 = rand_field(1, (8,), F)
    decomp = lambda s: s + s
    txt = jax.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
    assert txt.count("stablehlo.composite") == 1, txt
    assert "zorch.round" in txt


if __name__ == "__main__":
    test_runs_the_decomposition()
    if _HAS_COMPOSITE_OP:
        test_emits_one_zorch_round_composite()
    print("ok")
