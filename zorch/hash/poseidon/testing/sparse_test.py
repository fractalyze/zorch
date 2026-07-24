# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Optimized-sparse Poseidon: byte-matches a pure-Python reference, emits a marker.

A small width-4 config over a 31-bit field pins the round structure — initial
ARC, the full/transition/partial/full/final split, the dense MDS in full rounds,
the transition matrix `P`, and the partial round's sparse lane-0-dot + lane-t
rank-1 update — against an independent numpy/Python reference. The constants are
arbitrary (not a real optimized instance); the test validates the *schedule
wiring*, while the real optimized-Poseidon byte-match lives with the reference
that supplies genuine constants.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F
from zk_dtypes import pfinfo

from zorch.fusion import FUSED_REGION_MARKER
from zorch.hash.permutation import Permutation
from zorch.hash.poseidon.params import SparsePoseidonParams
from zorch.hash.poseidon.sparse import SparsePoseidon

_P = pfinfo(F).modulus  # field prime; canonical-int reference reduces mod this.

# A small width-4 config. alpha=7 is coprime to p-1 for this field, so the S-box
# is a permutation. half_full_rounds=2 -> one pre-partial and one post-partial
# full round plus the transition and final rounds; 2 partial rounds.
_WIDTH, _HALF, _NPART, _ALPHA = 4, 2, 2, 7

_INITIAL_ARC = (3, 5, 7, 11)
_FULL_RC_PRE = ((13, 17, 19, 23),)  # (half-1, width)
_TRANSITION_RC = (29, 31, 37, 41)
_PARTIAL_RC = (43, 47)  # (n_partial,)
_FULL_RC_POST = ((53, 59, 61, 67),)  # (half-1, width)
_MDS = ((2, 3, 1, 4), (1, 2, 3, 1), (4, 1, 2, 3), (3, 4, 1, 2))
_TRANSITION_M = ((5, 1, 2, 1), (1, 6, 1, 2), (2, 1, 7, 1), (1, 2, 1, 8))
_PARTIAL_DOT = ((9, 2, 3, 5), (7, 1, 4, 6))  # (n_partial, width)
_PARTIAL_COL = ((2, 3, 5), (1, 4, 2))  # (n_partial, width-1)


def _to_field(canon: np.ndarray) -> fnp.ndarray:
    return fnp.asarray(canon.astype(np.int64).astype(F))


def _to_canon(arr: fnp.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(arr).astype(object), dtype=object)


def _params() -> SparsePoseidonParams:
    def fld(rows: object) -> fnp.ndarray:
        return _to_field(np.array(rows, dtype=np.int64))

    return SparsePoseidonParams(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        half_full_rounds=_HALF,
        n_partial_rounds=_NPART,
        initial_arc=fld(_INITIAL_ARC),
        full_rc_pre=fld(_FULL_RC_PRE),
        transition_rc=fld(_TRANSITION_RC),
        partial_rc=fld(_PARTIAL_RC),
        full_rc_post=fld(_FULL_RC_POST),
        mds=fld(_MDS),
        transition_matrix=fld(_TRANSITION_M),
        partial_dot=fld(_PARTIAL_DOT),
        partial_col=fld(_PARTIAL_COL),
    )


def _reference_permute(state_canon: list[int]) -> list[int]:
    """Independent optimized-sparse Poseidon reference in pure-Python int mod p.

    Mirrors the documented schedule: initial ARC, (half-1) full rounds with the
    dense MDS, one transition round with `P`, n_partial sparse rounds (S-box on
    lane 0, then lane-0 dot + lane-t rank-1 update), (half-1) full rounds, and a
    final full round with no trailing constant. The S-box precedes the round
    constant (the constant seeds the next round's S-box); the initial ARC seeds
    the first.
    """
    p, w, alpha = _P, _WIDTH, _ALPHA
    s = [x % p for x in state_canon]

    def sbox(x: int) -> int:
        return pow(x % p, alpha, p)

    def matmul(mat: Sequence[Sequence[int]], vec: list[int]) -> list[int]:
        return [sum(mat[i][j] * vec[j] for j in range(w)) % p for i in range(w)]

    def full_round(
        vec: list[int], rc: Sequence[int], mat: Sequence[Sequence[int]]
    ) -> list[int]:
        return matmul(mat, [(sbox(vec[i]) + rc[i]) % p for i in range(w)])

    s = [(s[i] + _INITIAL_ARC[i]) % p for i in range(w)]
    for r in range(_HALF - 1):
        s = full_round(s, _FULL_RC_PRE[r], _MDS)
    s = full_round(s, _TRANSITION_RC, _TRANSITION_M)
    for r in range(_NPART):
        a = (sbox(s[0]) + _PARTIAL_RC[r]) % p
        old = [a] + s[1:]
        new0 = sum(old[j] * _PARTIAL_DOT[r][j] for j in range(w)) % p
        rest = [(s[t] + a * _PARTIAL_COL[r][t - 1]) % p for t in range(1, w)]
        s = [new0] + rest
    for r in range(_HALF - 1):
        s = full_round(s, _FULL_RC_POST[r], _MDS)
    s = matmul(_MDS, [sbox(x) for x in s])
    return s


class SparsePoseidonReferenceByteMatchTest(absltest.TestCase):
    def test_byte_matches_reference(self) -> None:
        perm = SparsePoseidon(_params())
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _P, size=_WIDTH, dtype=np.int64)
            out = perm.permute(_to_field(canon))
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon])
            self.assertEqual(got, want)


class SparsePoseidonPermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        perm = SparsePoseidon(_params())
        self.assertIsInstance(perm, Permutation)
        self.assertEqual(perm.width, _WIDTH)
        self.assertEqual(perm.dtype, F)
        # No sparse-structure emitter: stays on the generic-marker per-block path.
        self.assertFalse(perm.has_dedicated_fusion)

    def test_permute_shape_and_vmap(self) -> None:
        perm = SparsePoseidon(_params())
        x = fnp.arange(_WIDTH, dtype=F)
        out = perm.permute(x)
        self.assertEqual(out.shape, (_WIDTH,))
        self.assertEqual(out.dtype, F)
        batch = fnp.stack([x, x + F(1)])
        bout = frx.vmap(perm.permute)(batch)
        self.assertEqual(bout.shape, (2, _WIDTH))
        self.assertTrue(bool(fnp.array_equal(bout[0], out)))

    def test_permute_rejects_wrong_shape(self) -> None:
        perm = SparsePoseidon(_params())
        with self.assertRaises(ValueError):
            perm.permute(fnp.zeros((_WIDTH + 1,), dtype=F))
        with self.assertRaises(ValueError):
            perm.permute(fnp.zeros((2, _WIDTH), dtype=F))


class SparsePoseidonParamsValidationTest(absltest.TestCase):
    def test_value_equality_and_hash(self) -> None:
        self.assertEqual(_params(), _params())
        self.assertEqual(hash(_params()), hash(_params()))

    def test_rejects_wrong_partial_col_shape(self) -> None:
        def fld(rows: object) -> fnp.ndarray:
            return _to_field(np.array(rows, dtype=np.int64))

        with self.assertRaises(ValueError):
            SparsePoseidonParams(
                width=_WIDTH,
                dtype=F,
                alpha=_ALPHA,
                half_full_rounds=_HALF,
                n_partial_rounds=_NPART,
                initial_arc=fld(_INITIAL_ARC),
                full_rc_pre=fld(_FULL_RC_PRE),
                transition_rc=fld(_TRANSITION_RC),
                partial_rc=fld(_PARTIAL_RC),
                full_rc_post=fld(_FULL_RC_POST),
                mds=fld(_MDS),
                transition_matrix=fld(_TRANSITION_M),
                partial_dot=fld(_PARTIAL_DOT),
                partial_col=fld(((2, 3, 5, 9), (1, 4, 2, 8))),  # width, not width-1
            )


class SparsePoseidonMarkerEmissionTest(absltest.TestCase):
    def test_permute_emits_generic_fused_region(self) -> None:
        # Dense full-field matrices (not int-literal-carryable), so — like
        # Poseidon2's free-form path — the permute marks its region with the
        # generic "zorch.fused_region" name; the normal-form body still fuses.
        perm = SparsePoseidon(_params())
        txt = frx.jit(perm.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{FUSED_REGION_MARKER}"', composite_line)


if __name__ == "__main__":
    absltest.main()
