"""permute lowers with no reduce/gather/dot (fusion-shaped by construction)."""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_params


def test_no_reduce_gather_dot_in_stablehlo():
    p = Poseidon2(koalabear16_params())
    text = jax.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
    for banned in ("stablehlo.reduce", "stablehlo.gather", "stablehlo.dot"):
        assert banned not in text, f"{banned} present — linear layer is not normal-form"


if __name__ == "__main__":
    test_no_reduce_gather_dot_in_stablehlo()
    print("ok")
