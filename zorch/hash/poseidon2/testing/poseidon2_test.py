"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    koalabear16_perm,
)


class Poseidon2Koalabear16Test(absltest.TestCase):
    def test_permute_byte_matches_plonky3(self) -> None:
        p = koalabear16_perm()
        out = p.permute(jnp.arange(16, dtype=F))
        self.assertTrue(bool(jnp.array_equal(out, KOALABEAR16_EXPECTED)))

    def test_vmap_batch_matches(self) -> None:
        p = koalabear16_perm()
        x = jnp.arange(16, dtype=F)
        batch = jax.vmap(p.permute)(jnp.stack([x, x]))
        self.assertTrue(bool(jnp.array_equal(batch[0], KOALABEAR16_EXPECTED)))
        self.assertTrue(bool(jnp.array_equal(batch[1], KOALABEAR16_EXPECTED)))


if __name__ == "__main__":
    absltest.main()
