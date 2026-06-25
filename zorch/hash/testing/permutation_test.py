"""Permutation Protocol is structural and hash-agnostic."""

from __future__ import annotations

import jax.numpy as jnp
from absl.testing import absltest
from jax import Array

from zorch.hash.permutation import Permutation


class _Id:
    width = 3
    dtype = jnp.int32
    has_dedicated_fusion = False  # no dedicated marker -> consumers use a fallback

    def permute(self, state: Array) -> Array:
        return state


class PermutationProtocolTest(absltest.TestCase):
    def test_duck_typed_impl_satisfies_protocol(self) -> None:
        self.assertIsInstance(_Id(), Permutation)

    def test_consumer_reads_width_and_dtype_without_naming_a_hash(self) -> None:
        p = _Id()
        state = jnp.zeros(p.width, dtype=p.dtype)  # sponge-style allocation
        self.assertEqual(state.shape, (3,))
        self.assertTrue(bool(jnp.array_equal(p.permute(state), state)))


if __name__ == "__main__":
    absltest.main()
