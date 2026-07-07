"""Permutation Protocol is structural and hash-agnostic."""

from __future__ import annotations

from collections.abc import Callable

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

    # Inert fused-region ABI — required by the seam, never called for a non-fused
    # permutation (`has_dedicated_fusion` False).
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, object]]:
        return (leading,), (lambda state, *ops: self.permute(state)), {}


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
