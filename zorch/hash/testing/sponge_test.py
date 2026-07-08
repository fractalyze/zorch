"""Sponge over a Permutation — contract + correctness (padding-free + Merkle-Damgard).

The koalabear-16 poseidon2 is the golden permutation (byte-matches Plonky3
4318eba). The expected outputs below replay the padding-free absorb step by step
(replace state[:rate], permute; a partial final block overwrites only its lanes,
no padding) against that golden permutation — pinning block boundaries, absorb
semantics, and the partial-block rule. An independent Plonky3-generated sponge
vector is added in the golden-vector slice. The construction-level Merkle-Damgard
tests live here too, since the construction is the Sponge's, not a permutation's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import export
from zk_dtypes import koalabear_mont as F

from zorch.hash.permutation import Permutation
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import SPONGE_HASH_MARKER, Sponge, SpongeParams, SpongeType

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


class SpongeTest(absltest.TestCase):
    def test_hash_returns_out_shape_and_dtype(self) -> None:
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        out = s.hash(jnp.arange(16, dtype=F))
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, F)

    def test_hash_single_block_is_permute_truncated(self) -> None:
        perm = koalabear16_perm()
        s = Sponge(perm, SpongeParams(rate=8, out=8))
        x = jnp.arange(8, dtype=F)  # exactly one rate block
        expected = perm.permute(jnp.zeros(16, dtype=F).at[:8].set(x))[:8]
        self.assertTrue(bool(jnp.array_equal(s.hash(x), expected)))

    def test_hash_two_full_blocks_overwrite_mode(self) -> None:
        perm = koalabear16_perm()
        s = Sponge(perm, SpongeParams(rate=8, out=8))
        x = jnp.arange(16, dtype=F)  # two rate blocks
        st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
        st = perm.permute(st)
        st = st.at[:8].set(x[8:16])  # overwrite (not XOR) the rate lanes
        st = perm.permute(st)
        self.assertTrue(bool(jnp.array_equal(s.hash(x), st[:8])))

    def test_hash_partial_final_block_overwrites_only_its_lanes(self) -> None:
        perm = koalabear16_perm()
        s = Sponge(perm, SpongeParams(rate=8, out=8))
        x = jnp.arange(12, dtype=F)  # rate + 4: final block is partial
        st = jnp.zeros(16, dtype=F).at[:8].set(x[:8])
        st = perm.permute(st)
        st = st.at[:4].set(x[8:12])  # only 4 lanes overwritten; lanes 4..7 keep value
        st = perm.permute(st)
        self.assertTrue(bool(jnp.array_equal(s.hash(x), st[:8])))

    def test_rate_not_less_than_width_raises(self) -> None:
        perm = koalabear16_perm()
        with self.assertRaises(ValueError):
            Sponge(perm, SpongeParams(rate=16, out=8))

    def test_invalid_params_raise(self) -> None:
        for rate, out in ((0, 8), (8, 0)):  # rate < 1, out < 1
            with self.assertRaises(ValueError):
                SpongeParams(rate=rate, out=out)

    def test_hash_non_1d_input_raises(self) -> None:
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        with self.assertRaises(ValueError):
            s.hash(jnp.arange(16, dtype=F).reshape(2, 8))  # 2-D, not 1-D

    def test_hash_matches_plonky3_golden(self) -> None:
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        for n, golden in _PLONKY3_SPONGE.items():
            self.assertTrue(
                bool(jnp.array_equal(s.hash(jnp.arange(n, dtype=F)), golden)),
                f"len {n}",
            )

    def test_hash_many_blocks_matches_stepwise_replay(self) -> None:
        # n=28/32 drive the absorb through 2-3 blocks (carry across blocks)
        # with and without a partial tail — regimes the fixed vectors above
        # never reach.
        perm = koalabear16_perm()
        s = Sponge(perm, SpongeParams(rate=8, out=8))
        for n in (28, 32):
            x = jnp.arange(n, dtype=F)
            st = jnp.zeros(16, dtype=F)
            for start in range(0, n, 8):
                block = x[start : start + 8]
                st = st.at[: block.shape[0]].set(block)
                st = perm.permute(st)
            self.assertTrue(bool(jnp.array_equal(s.hash(x), st[:8])), f"len {n}")

    def test_value_equality_across_fresh_instances(self) -> None:
        # A sponge seats in static jit-zone keys (#214): equal params over
        # value-equal permutations must compare and hash equal regardless of
        # instance identity, and a param change must break equality.
        a = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        b = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, Sponge(koalabear16_perm(), SpongeParams(rate=4, out=8)))

    def test_hash_vmap_matches_unbatched(self) -> None:
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        a = jnp.arange(16, dtype=F)
        b = jnp.arange(16, dtype=F) + F(3)
        batched = jax.vmap(s.hash)(jnp.stack([a, b]))
        self.assertTrue(bool(jnp.array_equal(batched[0], s.hash(a))))
        self.assertTrue(bool(jnp.array_equal(batched[1], s.hash(b))))


def _ref_merkle_damgard(
    perm: Permutation, x: jnp.ndarray, rate: int, out: int
) -> jnp.ndarray:
    """Independent Merkle-Damgard reference: per-block unroll — zero-pad a short
    final block, chain the prior digest state[:out] into capacity [rate:rate+out].
    Permutation-agnostic, so it cross-checks Sponge.hash(MERKLE_DAMGARD) over any
    permutation."""
    w = perm.width
    n = int(x.shape[0])
    st = jnp.zeros(w, dtype=x.dtype)
    for blk in range((n + rate - 1) // rate):
        start = blk * rate
        count = min(rate, n - start)
        cap = st[:out]  # prior digest (zeros on block 0)
        st = st.at[:count].set(x[start : start + count])
        if count < rate:  # zero-pad the partial tail
            st = st.at[count:rate].set(jnp.zeros(rate - count, dtype=x.dtype))
        st = st.at[rate : rate + out].set(cap)  # chain
        st = perm.permute(st)
    return st[:out]


# Permutations the construction is exercised over — add a row to cover another
# (the Merkle-Damgard construction is the Sponge's, so one test serves all).
# (permutation, rate, out) with rate + out == width.
_MD_CASES = ((koalabear16_perm(), 8, 8),)


class MerkleDamgardTest(absltest.TestCase):
    """The Merkle-Damgard construction lives in Sponge, so it is tested here once
    (over any permutation) rather than per hash."""

    def test_matches_stepwise_reference(self) -> None:
        for perm, rate, out in _MD_CASES:
            s = Sponge(perm, SpongeParams(rate=rate, out=out))
            # one full block, two full, then two partial-tail lengths.
            for n in (rate, 2 * rate, rate + rate // 2, 2 * rate + rate // 2):
                x = jnp.arange(n, dtype=F)
                got = s.hash(x, sponge_type=SpongeType.MERKLE_DAMGARD)
                self.assertTrue(
                    bool(jnp.array_equal(got, _ref_merkle_damgard(perm, x, rate, out))),
                    f"width {perm.width}, len {n}",
                )

    def test_requires_rate_plus_out_equals_width(self) -> None:
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=4))  # 8 + 4 != 16
        with self.assertRaises(ValueError):
            s.hash(jnp.arange(8, dtype=F), sponge_type=SpongeType.MERKLE_DAMGARD)

    def test_lowers_under_symbolic_length(self) -> None:
        # Rides the shared while_loop absorb, so a symbolic `len(input)` lowers
        # (emits the sponge_hash marker) rather than needing a static-n path.
        s = Sponge(koalabear16_perm(), SpongeParams(rate=8, out=8))
        (n,) = export.symbolic_shape("n")
        txt = (
            jax.jit(lambda x: s.hash(x, sponge_type=SpongeType.MERKLE_DAMGARD))
            .lower(jax.ShapeDtypeStruct((n,), F))
            .as_text()
        )
        self.assertIn(f'"{SPONGE_HASH_MARKER}"', txt)


if __name__ == "__main__":
    absltest.main()
