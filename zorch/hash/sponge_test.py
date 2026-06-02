"""Sponge (padding-free, overwrite-mode) over a Permutation — contract + correctness.

The koalabear-16 poseidon2 is the golden permutation (byte-matches Plonky3
4318eba). The expected outputs below replay the overwrite-mode absorb step by
step (replace state[:rate], permute; a partial final block overwrites only its
lanes, no padding) against that golden permutation — pinning block boundaries,
overwrite semantics, and the partial-block rule. An independent Plonky3-generated
sponge vector is added in the golden-vector slice.
"""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams

# Plonky3 golden vectors (p3_commit=4318eba..., default_koalabear_poseidon2_16):
# PaddingFreeSponge<_, 16, 8, 8> over arange(n). arange(12) exercises the
# partial final block (overwrite-mode, no padding).
_PLONKY3_SPONGE = {
    8: jnp.array(
        [
            966837595,
            1699035679,
            1922113316,
            2023906830,
            809021653,
            1488168764,
            817789182,
            1446614690,
        ],
        dtype=F,
    ),
    16: jnp.array(
        [
            2093465132,
            1938411931,
            339653216,
            887899196,
            588109084,
            84644453,
            365237898,
            96671732,
        ],
        dtype=F,
    ),
    12: jnp.array(
        [
            1283734044,
            275672105,
            632916173,
            1607999122,
            756879617,
            175064997,
            961395546,
            931537840,
        ],
        dtype=F,
    ),
}


def test_hash_returns_out_shape_and_dtype():
    s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
    out = s.hash(jnp.arange(16, dtype=F))
    assert out.shape == (8,)
    assert out.dtype == F


def test_hash_single_block_is_permute_truncated():
    perm = koalabear16_perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(8, dtype=F)  # exactly one rate block
    expected = perm.permute(jnp.zeros(16, dtype=F).at[:8].set(x))[:8]
    assert jnp.array_equal(s.hash(x), expected)


def test_hash_two_full_blocks_overwrite_mode():
    perm = koalabear16_perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(16, dtype=F)  # two rate blocks
    st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
    st = perm.permute(st)
    st = st.at[:8].set(x[8:16])  # overwrite (not XOR) the rate lanes
    st = perm.permute(st)
    assert jnp.array_equal(s.hash(x), st[:8])


def test_hash_partial_final_block_overwrites_only_its_lanes():
    perm = koalabear16_perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(12, dtype=F)  # rate + 4: final block is partial
    st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
    st = perm.permute(st)
    st = st.at[:4].set(x[8:12])  # only 4 lanes overwritten; lanes 4..7 keep value
    st = perm.permute(st)
    assert jnp.array_equal(s.hash(x), st[:8])


def test_rate_not_less_than_width_raises():
    perm = koalabear16_perm()
    try:
        Sponge(perm, SpongeParams(rate=16, out=8))
        assert False, "expected ValueError for rate >= width"
    except ValueError:
        pass


def test_invalid_params_raise():
    for rate, out in ((0, 8), (8, 0)):  # rate < 1, out < 1
        try:
            SpongeParams(rate=rate, out=out)
            assert False, f"expected ValueError for rate={rate}, out={out}"
        except ValueError:
            pass


def test_hash_non_1d_input_raises():
    s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
    try:
        s.hash(jnp.arange(16, dtype=F).reshape(2, 8))  # 2-D, not 1-D
        assert False, "expected ValueError for non-1-D input"
    except ValueError:
        pass


def test_hash_matches_plonky3_golden():
    s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
    for n, golden in _PLONKY3_SPONGE.items():
        assert jnp.array_equal(s.hash(jnp.arange(n, dtype=F)), golden), f"len {n}"


def test_hash_vmap_matches_unbatched():
    s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
    a = jnp.arange(16, dtype=F)
    b = jnp.arange(16, dtype=F) + F(3)
    batched = jax.vmap(s.hash)(jnp.stack([a, b]))
    assert jnp.array_equal(batched[0], s.hash(a))
    assert jnp.array_equal(batched[1], s.hash(b))


if __name__ == "__main__":
    test_hash_returns_out_shape_and_dtype()
    test_hash_single_block_is_permute_truncated()
    test_hash_two_full_blocks_overwrite_mode()
    test_hash_partial_final_block_overwrites_only_its_lanes()
    test_rate_not_less_than_width_raises()
    test_invalid_params_raise()
    test_hash_non_1d_input_raises()
    test_hash_matches_plonky3_golden()
    test_hash_vmap_matches_unbatched()
    print("ok")
