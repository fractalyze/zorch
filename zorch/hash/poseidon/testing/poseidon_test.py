"""Classic Poseidon: byte-matches a pure-Python reference and emits its marker.

A small width-3 config over a 31-bit field is enough to pin the round structure
(full/partial split, ARC every round, dense MDS every round, S-box on all vs the
last lane) against an independent numpy/Python reference, and to read the marker
the dedicated emitter consumes off the lowered StableHLO.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F
from zk_dtypes import pfinfo

from zorch.hash.permutation import Permutation
from zorch.hash.poseidon.params import PoseidonParams
from zorch.hash.poseidon.poseidon import (
    POSEIDON_MARKER,
    POSEIDON_MARKER_VERSION,
    Poseidon,
)
from zorch.hash.sponge import Sponge, SpongeParams

_P = pfinfo(F).modulus  # field prime; canonical-int reference reduces mod this.

# A small width-3 config. alpha=7 is coprime to p-1 for this field, so the
# S-box is a permutation; the MDS is a small-int MDS matrix.
_WIDTH, _FULL, _PARTIAL, _ALPHA = 3, 2, 1, 7
_MDS = ((2, 3, 1), (1, 2, 3), (3, 1, 2))
# One full-width ARC vector per round (full_rounds + partial_rounds rows).
_ROUND_CONSTANTS = (
    (5, 7, 11),
    (13, 17, 19),  # the single partial round
    (23, 29, 31),
)

# Marker metadata as StableHLO prints it (dict keys alphabetical). `mds` is the
# 3x3 MDS flattened row-major, a numpy int64 value so it lowers to a
# DenseElementsAttr the zkx recognizer parses (GetCompositeAttrIntArray).
EXPECTED_ATTRS = (
    f"composite_attributes = {{alpha = {_ALPHA} : i64,"
    f" full_rounds = {_FULL} : i64,"
    " mds = dense<[2, 3, 1, 1, 2, 3, 3, 1, 2]> : tensor<9xi64>,"
    f" partial_rounds = {_PARTIAL} : i64, width = {_WIDTH} : i64}}"
)


def _to_field(canon: np.ndarray) -> jnp.ndarray:
    """Canonical ints -> field array (the dtype cast Montgomery-encodes)."""
    return jnp.asarray(canon.astype(np.int64).astype(F))


def _to_canon(arr: jnp.ndarray) -> np.ndarray:
    """Field array -> canonical ints (numpy object cast Montgomery-decodes,
    no jax x64 needed)."""
    return np.asarray(np.asarray(arr).astype(object), dtype=object)


def _poseidon_params() -> PoseidonParams:
    rc = np.array(_ROUND_CONSTANTS, dtype=np.int64)
    mds = np.array(_MDS, dtype=np.int64)
    return PoseidonParams(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        full_rounds=_FULL,
        partial_rounds=_PARTIAL,
        round_constants=_to_field(rc),
        mds=_to_field(mds),
    )


def _reference_permute(state_canon: list[int]) -> list[int]:
    """Independent classic-Poseidon reference in pure-Python int arithmetic mod p.

    Each round: ARC (add the round's full-width constants) -> S-box (x^alpha on
    all lanes in a full round, on the last lane only in a partial round) -> dense
    MDS (mds @ state). Rounds split full_rounds/2 full, partial_rounds partial,
    full_rounds/2 full; the dense MDS runs every round.
    """
    p, w, alpha = _P, _WIDTH, _ALPHA
    s = [x % p for x in state_canon]
    rounds = [list(r) for r in _ROUND_CONSTANTS]
    half_full = _FULL // 2

    def sbox(x: int) -> int:
        return pow(x % p, alpha, p)

    def mds(vec: list[int]) -> list[int]:
        return [sum(_MDS[i][j] * vec[j] for j in range(w)) % p for i in range(w)]

    r = 0
    for _ in range(half_full):  # initial full rounds
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s = [sbox(x) for x in s]
        s = mds(s)
        r += 1
    for _ in range(_PARTIAL):  # partial rounds: S-box on the last lane only
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s[w - 1] = sbox(s[w - 1])
        s = mds(s)
        r += 1
    for _ in range(half_full):  # terminal full rounds
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s = [sbox(x) for x in s]
        s = mds(s)
        r += 1
    return s


class PoseidonReferenceByteMatchTest(absltest.TestCase):
    """Poseidon.permute byte-matches an independent pure-Python reference over
    random inputs (run eagerly on CPU)."""

    def test_byte_matches_reference(self) -> None:
        p = Poseidon(_poseidon_params())
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _P, size=_WIDTH, dtype=np.int64)
            state = _to_field(canon)
            out = p.permute(state)
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon])
            self.assertEqual(got, want)


class PoseidonPermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        p = Poseidon(_poseidon_params())
        self.assertIsInstance(p, Permutation)
        self.assertEqual(p.width, _WIDTH)
        self.assertEqual(p.dtype, F)
        self.assertTrue(p.has_dedicated_fusion)

    def test_permute_shape_and_vmap(self) -> None:
        p = Poseidon(_poseidon_params())
        x = jnp.arange(_WIDTH, dtype=F)
        out = p.permute(x)
        self.assertEqual(out.shape, (_WIDTH,))
        self.assertEqual(out.dtype, F)
        batch = jnp.stack([x, x + F(1)])
        bout = jax.vmap(p.permute)(batch)  # thread-per-hash
        self.assertEqual(bout.shape, (2, _WIDTH))
        self.assertEqual(bout.dtype, F)
        self.assertTrue(bool(jnp.array_equal(bout[0], out)))

    def test_permute_rejects_wrong_shape(self) -> None:
        p = Poseidon(_poseidon_params())
        with self.assertRaises(ValueError):
            p.permute(jnp.zeros((_WIDTH + 1,), dtype=F))  # width mismatch
        with self.assertRaises(ValueError):
            p.permute(jnp.zeros((2, _WIDTH), dtype=F))  # batched, not a 1-D state


class PoseidonMarkerEmissionTest(absltest.TestCase):
    def test_permute_emits_poseidon_named_composite(self) -> None:
        # The permute marks its region "zorch.poseidon" so zkx routes it to the
        # dedicated Poseidon emitter; the permutation shape rides as
        # composite.attributes — all four ints plus the flat MDS are required by
        # the zkx recognizer.
        p = Poseidon(_poseidon_params())
        txt = jax.jit(p.permute).lower(jnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON_MARKER}"', composite_line)
        self.assertIn(EXPECTED_ATTRS, composite_line)
        self.assertIn(f"version = {POSEIDON_MARKER_VERSION}", composite_line)
        # Exactly the 2 ABI operands [state, round_constants flattened]. A
        # closed-over MDS would be lifted to a leading operand (jax.lax.composite
        # prepends consts) and break the 2-operand Poseidon emitter ABI.
        operands = composite_line.split(f'"{POSEIDON_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 2, composite_line)

    def test_mds_serializes_as_dense_i64_tensor(self) -> None:
        # The mds attribute must lower to a DenseElementsAttr
        # (`dense<[..]> : tensor<Nxi64>`), the form the zkx recognizer reads via
        # GetCompositeAttrIntArray — NOT a plain ArrayAttr (`mds = [..]`), which a
        # Python list/tuple would produce.
        p = Poseidon(_poseidon_params())
        txt = jax.jit(p.permute).lower(jnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertIn("mds = dense<[2, 3, 1, 1, 2, 3, 3, 1, 2]> : tensor<9xi64>", txt)


def _ref_chained(p: Poseidon, x: jnp.ndarray, rate: int, out: int) -> jnp.ndarray:
    """Independent Merkle-Damgard reference (zero-pad short block; chain the prior
    digest state[:out] into capacity [rate:rate+out]). Classic-Poseidon mirror of
    the Poseidon2 chained test — pins the reusability win (chained over any
    permutation, from the shared absorb)."""
    w = p.width
    n = int(x.shape[0])
    st = jnp.zeros(w, dtype=x.dtype)
    for blk in range((n + rate - 1) // rate):
        start = blk * rate
        count = min(rate, n - start)
        cap = st[:out]
        st = st.at[:count].set(x[start : start + count])
        if count < rate:
            st = st.at[count:rate].set(jnp.zeros(rate - count, dtype=x.dtype))
        st = st.at[rate : rate + out].set(cap)
        st = p.permute(st)
    return st[:out]


class PoseidonChainedHashTest(absltest.TestCase):
    def test_linear_hash_matches_stepwise_merkle_damgard(self) -> None:
        # Classic Poseidon gets linear_hash for free (shared absorb). width 3,
        # rate 2 + out 1 == width. n=2 one block, 4 two full, 3/5 partial tail.
        p = Poseidon(_poseidon_params())
        s = Sponge(p, SpongeParams(rate=2, out=1))
        for n in (2, 3, 4, 5):
            x = jnp.arange(n, dtype=F)
            self.assertTrue(
                bool(jnp.array_equal(s.linear_hash(x), _ref_chained(p, x, 2, 1))),
                f"len {n}",
            )

    def test_linear_hash_requires_rate_plus_out_equals_width(self) -> None:
        p = Poseidon(_poseidon_params())  # width 3
        s = Sponge(p, SpongeParams(rate=2, out=2))  # 2 + 2 != 3
        with self.assertRaises(ValueError):
            s.linear_hash(jnp.arange(2, dtype=F))


if __name__ == "__main__":
    absltest.main()
