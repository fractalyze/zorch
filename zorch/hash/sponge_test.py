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

from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_params
from zorch.hash.sponge import Sponge, SpongeParams


def _perm() -> Poseidon2:
    return Poseidon2(koalabear16_params())  # width 16


def test_hash_returns_out_shape_and_dtype():
    s = Sponge(_perm(), SpongeParams(rate=8, out=8))
    out = s.hash(jnp.arange(16, dtype=F))
    assert out.shape == (8,)
    assert out.dtype == F


def test_hash_single_block_is_permute_truncated():
    perm = _perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(8, dtype=F)  # exactly one rate block
    expected = perm.permute(jnp.zeros(16, dtype=F).at[:8].set(x))[:8]
    assert jnp.array_equal(s.hash(x), expected)


def test_hash_two_full_blocks_overwrite_mode():
    perm = _perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(16, dtype=F)  # two rate blocks
    st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
    st = perm.permute(st)
    st = st.at[:8].set(x[8:16])  # overwrite (not XOR) the rate lanes
    st = perm.permute(st)
    assert jnp.array_equal(s.hash(x), st[:8])


def test_hash_partial_final_block_overwrites_only_its_lanes():
    perm = _perm()
    s = Sponge(perm, SpongeParams(rate=8, out=8))
    x = jnp.arange(12, dtype=F)  # rate + 4: final block is partial
    st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
    st = perm.permute(st)
    st = st.at[:4].set(x[8:12])  # only 4 lanes overwritten; lanes 4..7 keep value
    st = perm.permute(st)
    assert jnp.array_equal(s.hash(x), st[:8])


def test_rate_not_less_than_width_raises():
    perm = _perm()  # width 16
    try:
        Sponge(perm, SpongeParams(rate=16, out=8))
        assert False, "expected ValueError for rate >= width"
    except ValueError:
        pass


def test_hash_vmap_matches_unbatched():
    s = Sponge(_perm(), SpongeParams(rate=8, out=8))
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
    test_hash_vmap_matches_unbatched()
    print("ok")
