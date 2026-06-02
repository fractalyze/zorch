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
    # arity*chunk (8) < width (16): inputs go in the first lanes, rest stays zero.
    perm = _perm()
    c = Compression(perm, CompressionParams(arity=2, chunk=4))
    x = jnp.arange(8, dtype=F).reshape(2, 4)
    pre = jnp.zeros(perm.width, dtype=F).at[:8].set(jnp.arange(8, dtype=F))
    expected = perm.permute(pre)[:4]
    assert jnp.array_equal(c.compress(x), expected)


def test_arity_chunk_exceeding_width_raises():
    perm = _perm()  # width 16
    try:
        Compression(perm, CompressionParams(arity=3, chunk=8))  # 24 > 16
        assert False, "expected ValueError for arity*chunk > width"
    except ValueError:
        pass


def test_invalid_params_raise():
    for arity, chunk in ((1, 8), (2, 0)):  # arity < 2, chunk < 1
        try:
            CompressionParams(arity=arity, chunk=chunk)
            assert False, f"expected ValueError for arity={arity}, chunk={chunk}"
        except ValueError:
            pass


def test_compress_wrong_input_shape_raises():
    c = Compression(_perm(), CompressionParams(arity=2, chunk=8))
    try:
        c.compress(jnp.arange(16, dtype=F))  # flat, not (2, 8)
        assert False, "expected ValueError for wrong input shape"
    except ValueError:
        pass


def test_compress_matches_plonky3_golden():
    c = Compression(_perm(), CompressionParams(arity=2, chunk=8))
    out = c.compress(jnp.arange(16, dtype=F).reshape(2, 8))
    assert jnp.array_equal(out, _PLONKY3_COMPRESS_2X8)


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
    test_invalid_params_raise()
    test_compress_wrong_input_shape_raises()
    test_compress_matches_plonky3_golden()
    test_compress_vmap_matches_unbatched()
    print("ok")
