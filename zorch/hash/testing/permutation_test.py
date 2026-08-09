"""Permutation Protocol is structural and hash-agnostic."""

from __future__ import annotations

from collections.abc import Callable

import frx.numpy as fnp
from absl.testing import absltest
from frx import Array

from zorch.fusion import FUSED_REGION_MARKER
from zorch.hash.permutation import Permutation


class _Id:
    width = 3
    dtype = fnp.int32
    has_dedicated_fusion = False  # no dedicated marker -> consumers use a fallback
    fused_region_marker = (FUSED_REGION_MARKER, 0)

    def permute(self, state: Array) -> Array:
        return state

    # Inert fused-region ABI: non-fused, never called.
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, object]]:
        return (leading,), (lambda state, *ops: self.permute(state)), {}


class PermutationProtocolTest(absltest.TestCase):
    def test_duck_typed_impl_satisfies_protocol(self) -> None:
        self.assertIsInstance(_Id(), Permutation)

    def test_consumer_reads_width_and_dtype_without_naming_a_hash(self) -> None:
        p = _Id()
        state = fnp.zeros(p.width, dtype=p.dtype)  # sponge-style allocation
        self.assertEqual(state.shape, (3,))
        self.assertTrue(bool(fnp.array_equal(p.permute(state), state)))


if __name__ == "__main__":
    absltest.main()
