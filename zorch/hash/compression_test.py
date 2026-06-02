"""Compression (truncated permutation) over a Permutation — contract + correctness.

The koalabear-16 poseidon2 is the golden permutation (byte-matches Plonky3
4318eba). Plonky3's `TruncatedPermutation::compress` is exactly
`permute(zero-pad(flatten(input)) to width)[:chunk]`, so matching that
composition on the golden permutation is the correctness oracle here — the tests
pin the packing (lane placement, flatten order, truncation) against an
independently built pre-image. An independent Plonky3-generated compression
vector is added in the golden-vector slice.
"""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_params


def _perm() -> Poseidon2:
    return Poseidon2(koalabear16_params())  # width 16


def test_compress_returns_chunk_shape_and_dtype():
    c = Compression(_perm(), CompressionParams(arity=2, chunk=8))
    out = c.compress(jnp.arange(16, dtype=F).reshape(2, 8))
    assert out.shape == (8,)
    assert out.dtype == F


def test_compress_2to1_is_full_width_permute_truncated():
    # arity*chunk == width: no padding; compress == permute(flatten)[:chunk].
    perm = _perm()
    c = Compression(perm, CompressionParams(arity=2, chunk=8))
    x = jnp.arange(16, dtype=F).reshape(2, 8)
    expected = perm.permute(x.reshape(-1))[:8]
    assert jnp.array_equal(c.compress(x), expected)


def test_compress_zero_pads_when_below_width():
    # arity*chunk < width: input goes in the first lanes, rest stays zero.
    perm = _perm()
    c = Compression(perm, CompressionParams(arity=1, chunk=4))
    x = jnp.arange(4, dtype=F).reshape(1, 4)
    pre = jnp.zeros(perm.width, dtype=F).at[:4].set(jnp.arange(4, dtype=F))
    expected = perm.permute(pre)[:4]
    assert jnp.array_equal(c.compress(x), expected)


def test_arity_chunk_exceeding_width_raises():
    perm = _perm()  # width 16
    try:
        Compression(perm, CompressionParams(arity=3, chunk=8))  # 24 > 16
        assert False, "expected ValueError for arity*chunk > width"
    except ValueError:
        pass


def test_compress_vmap_matches_unbatched():
    c = Compression(_perm(), CompressionParams(arity=2, chunk=8))
    a = jnp.arange(16, dtype=F).reshape(2, 8)
    b = (jnp.arange(16, dtype=F) + F(7)).reshape(2, 8)
    batched = jax.vmap(c.compress)(jnp.stack([a, b]))
    assert jnp.array_equal(batched[0], c.compress(a))
    assert jnp.array_equal(batched[1], c.compress(b))


if __name__ == "__main__":
    test_compress_returns_chunk_shape_and_dtype()
    test_compress_2to1_is_full_width_permute_truncated()
    test_compress_zero_pads_when_below_width()
    test_arity_chunk_exceeding_width_raises()
    test_compress_vmap_matches_unbatched()
    print("ok")
