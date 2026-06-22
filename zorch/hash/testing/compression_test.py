"""Compression (truncated permutation) over a Permutation — contract + correctness.

The koalabear-16 poseidon2 is the golden permutation (byte-matches Plonky3
4318eba). Plonky3's `TruncatedPermutation::compress` is exactly
`permute(zero-pad(flatten(input)) to width)[:chunk]`, so matching that
composition on the golden permutation is the correctness oracle here — the tests
pin the packing (lane placement, flatten order, truncation) against an
independently built pre-image. An independent Plonky3-generated compression
vector is added in the golden-vector slice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm

# Plonky3 golden vector (p3_commit=4318eba..., default_koalabear_poseidon2_16):
# TruncatedPermutation<_, 2, 8, 16> compressing arange(16) as two chunks of 8.
_PLONKY3_COMPRESS_2X8 = jnp.array(
    [
        1259554834,
        663463928,
        1989430097,
        476523442,
        836740795,
        1803459961,
        1229318262,
        2023956904,
    ],
    dtype=F,
)


class CompressionTest(absltest.TestCase):
    def test_compress_returns_chunk_shape_and_dtype(self) -> None:
        c = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        out = c.compress(jnp.arange(16, dtype=F).reshape(2, 8))
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, F)

    def test_compress_batched_matches_vmap(self) -> None:
        # compress_batched folds a whole sibling level at once; numerically a
        # vmap(compress) (it routes through permute_batched for the shared-body
        # lowering).
        c = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        groups = jnp.arange(7 * 2 * 8, dtype=F).reshape(7, 2, 8)
        self.assertTrue(
            bool(
                jnp.array_equal(
                    c.compress_batched(groups), jax.vmap(c.compress)(groups)
                )
            )
        )

    def test_compress_2to1_is_full_width_permute_truncated(self) -> None:
        # arity*chunk == width: no padding; compress == permute(flatten)[:chunk].
        perm = koalabear16_perm()
        c = Compression(perm, CompressionParams(arity=2, chunk=8))
        x = jnp.arange(16, dtype=F).reshape(2, 8)
        expected = perm.permute(x.reshape(-1))[:8]
        self.assertTrue(bool(jnp.array_equal(c.compress(x), expected)))

    def test_compress_zero_pads_when_below_width(self) -> None:
        # arity*chunk (8) < width (16): inputs go in the first lanes, rest stays zero.
        perm = koalabear16_perm()
        c = Compression(perm, CompressionParams(arity=2, chunk=4))
        x = jnp.arange(8, dtype=F).reshape(2, 4)
        pre = jnp.zeros(perm.width, dtype=F).at[:8].set(jnp.arange(8, dtype=F))
        expected = perm.permute(pre)[:4]
        self.assertTrue(bool(jnp.array_equal(c.compress(x), expected)))

    def test_arity_chunk_exceeding_width_raises(self) -> None:
        perm = koalabear16_perm()
        with self.assertRaises(ValueError):
            Compression(perm, CompressionParams(arity=3, chunk=8))  # 24 > 16

    def test_invalid_params_raise(self) -> None:
        for arity, chunk in ((1, 8), (2, 0)):  # arity < 2, chunk < 1
            with self.assertRaises(ValueError):
                CompressionParams(arity=arity, chunk=chunk)

    def test_compress_wrong_input_shape_raises(self) -> None:
        c = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        with self.assertRaises(ValueError):
            c.compress(jnp.arange(16, dtype=F))  # flat, not (2, 8)

    def test_compress_matches_plonky3_golden(self) -> None:
        c = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        out = c.compress(jnp.arange(16, dtype=F).reshape(2, 8))
        self.assertTrue(bool(jnp.array_equal(out, _PLONKY3_COMPRESS_2X8)))

    def test_value_equality_across_fresh_instances(self) -> None:
        # A compressor seats in static jit-zone keys (#214): equal params over
        # value-equal permutations must compare and hash equal regardless of
        # instance identity, and a param change must break equality.
        a = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        b = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(
            a, Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=4))
        )

    def test_compress_vmap_matches_unbatched(self) -> None:
        c = Compression(koalabear16_perm(), CompressionParams(arity=2, chunk=8))
        a = jnp.arange(16, dtype=F).reshape(2, 8)
        b = (jnp.arange(16, dtype=F) + F(7)).reshape(2, 8)
        batched = jax.vmap(c.compress)(jnp.stack([a, b]))
        self.assertTrue(bool(jnp.array_equal(batched[0], c.compress(a))))
        self.assertTrue(bool(jnp.array_equal(batched[1], c.compress(b))))


if __name__ == "__main__":
    absltest.main()
