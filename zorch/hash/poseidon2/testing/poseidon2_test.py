"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    koalabear16_params,
)


def test_permute_byte_matches_plonky3():
    p = Poseidon2(koalabear16_params())
    out = p.permute(jnp.arange(16, dtype=F))
    assert jnp.array_equal(out, KOALABEAR16_EXPECTED)


def test_vmap_batch_matches():
    p = Poseidon2(koalabear16_params())
    x = jnp.arange(16, dtype=F)
    batch = jax.vmap(p.permute)(jnp.stack([x, x]))
    assert jnp.array_equal(batch[0], KOALABEAR16_EXPECTED)
    assert jnp.array_equal(batch[1], KOALABEAR16_EXPECTED)


if __name__ == "__main__":
    test_permute_byte_matches_plonky3()
    test_vmap_batch_matches()
    print("ok")
