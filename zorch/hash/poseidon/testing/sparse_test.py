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
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F
from zk_dtypes import koalabear_mont as G  # a distinct field, for dtype-guard tests
from zk_dtypes import pfinfo

from zorch.fusion import FUSED_REGION_MARKER
from zorch.hash.permutation import Permutation
from zorch.hash.poseidon import sparse as sparse_mod
from zorch.hash.poseidon.params import SparsePoseidonParams
from zorch.hash.poseidon.sparse import (
    POSEIDON_SPARSE_MARKER,
    POSEIDON_SPARSE_MARKER_VERSION,
    SparsePoseidon,
)

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


def _fld(rows: object) -> fnp.ndarray:
    return _to_field(np.array(rows, dtype=np.int64))


def _param_kwargs() -> dict:
    """The valid width-4 param kwargs; validation tests override one field."""
    return dict(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        half_full_rounds=_HALF,
        n_partial_rounds=_NPART,
        initial_arc=_fld(_INITIAL_ARC),
        full_rc_pre=_fld(_FULL_RC_PRE),
        transition_rc=_fld(_TRANSITION_RC),
        partial_rc=_fld(_PARTIAL_RC),
        full_rc_post=_fld(_FULL_RC_POST),
        mds=_fld(_MDS),
        transition_matrix=_fld(_TRANSITION_M),
        partial_dot=_fld(_PARTIAL_DOT),
        partial_col=_fld(_PARTIAL_COL),
    )


def _params() -> SparsePoseidonParams:
    return SparsePoseidonParams(**_param_kwargs())


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

    def test_permute_rejects_wrong_dtype(self) -> None:
        # A right-shaped state in the wrong field must hit the TypeError branch,
        # not silently permute in the other field.
        perm = SparsePoseidon(_params())
        with self.assertRaises(TypeError):
            perm.permute(fnp.zeros((_WIDTH,), dtype=G))


class SparsePoseidonParamsValidationTest(absltest.TestCase):
    def test_value_equality_and_hash(self) -> None:
        self.assertEqual(_params(), _params())
        self.assertEqual(hash(_params()), hash(_params()))

    def test_rejects_wrong_partial_col_shape(self) -> None:
        with self.assertRaises(ValueError):
            SparsePoseidonParams(
                **{**_param_kwargs(), "partial_col": _fld(((2, 3, 5, 9), (1, 4, 2, 8)))}
            )  # width, not width-1

    def test_rejects_dtype_mismatch(self) -> None:
        # An array in a different field than `dtype` must be rejected.
        wrong = fnp.asarray(np.array(_INITIAL_ARC, dtype=np.int64).astype(G))
        with self.assertRaises(ValueError):
            SparsePoseidonParams(**{**_param_kwargs(), "initial_arc": wrong})

    def test_rejects_bad_scalars(self) -> None:
        # Each scalar guard fires independently.
        for field, bad in (
            ("alpha", 0),  # must be positive
            ("width", 1),  # must be >= 2 (else partial_col is (npr, 0))
            ("half_full_rounds", 0),  # must be positive
            ("n_partial_rounds", -1),  # must be non-negative
        ):
            with self.assertRaises(ValueError, msg=field):
                SparsePoseidonParams(**{**_param_kwargs(), field: bad})


class SparsePoseidonMarkerEmissionTest(absltest.TestCase):
    def test_permute_emits_generic_fused_region(self) -> None:
        # Default (emitter not shipped, `_DEDICATED_EMITTER_AVAILABLE=False`): the
        # permute marks its region with the generic "zorch.fused_region" name so no
        # compile fails on an unknown composite; the normal-form body still fuses.
        # Flips to the dedicated marker below once the emitter is available.
        perm = SparsePoseidon(_params())
        self.assertFalse(perm.has_dedicated_fusion)
        txt = frx.jit(perm.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{FUSED_REGION_MARKER}"', composite_line)


def _dedicated_perm(params: SparsePoseidonParams | None = None) -> SparsePoseidon:
    """A SparsePoseidon built with the dedicated emitter forced available, so its
    permute routes to the `zorch.sparse_poseidon` marker. `_DEDICATED_EMITTER_-
    AVAILABLE` is read in `__init__`, so the patch must wrap construction; the
    instance then carries the dedicated name/attrs and lowers the same afterwards."""
    with mock.patch.object(sparse_mod, "_DEDICATED_EMITTER_AVAILABLE", True):
        return SparsePoseidon(params if params is not None else _params())


class SparsePoseidonDedicatedMarkerTest(absltest.TestCase):
    """The dormant dedicated path: verified by lowering (the emitter is not
    published, so these must not compile/run the marker) and by exercising the
    reference body directly (eager, no emitter needed)."""

    def test_permute_emits_dedicated_composite(self) -> None:
        # When the emitter is available the permute marks its region
        # "zorch.sparse_poseidon" so XLA routes it to SparsePoseidonFusion. The
        # ABI operands are exactly [state, initial_arc, full_rc_pre, transition_rc,
        # partial_rc, full_rc_post] = 6; a closed-over matrix would be lifted to a
        # leading operand (frx.lax.composite prepends consts) and break that ABI.
        perm = _dedicated_perm()
        self.assertTrue(perm.has_dedicated_fusion)
        txt = frx.jit(perm.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON_SPARSE_MARKER}"', composite_line)
        self.assertIn(f"version = {POSEIDON_SPARSE_MARKER_VERSION}", composite_line)
        operands = composite_line.split(f'"{POSEIDON_SPARSE_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 6, composite_line)

    def test_shape_and_matrix_attrs_serialize_as_dense_i64(self) -> None:
        # The schedule shape rides as int attrs and the four matrices as
        # DenseElementsAttrs (`dense<[..]> : tensor<Nxi64>`) — the form the XLA
        # recognizer reads via GetCompositeAttrIntArray, NOT a plain ArrayAttr
        # (`mds = [..]`) a Python list would produce. Matrices flatten row-major.
        perm = _dedicated_perm()
        txt = frx.jit(perm.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        for shape_attr in (
            f"width = {_WIDTH} : i64",
            f"half_full_rounds = {_HALF} : i64",
            f"n_partial_rounds = {_NPART} : i64",
            f"alpha = {_ALPHA} : i64",
        ):
            self.assertIn(shape_attr, txt)
        self.assertIn(
            "mds = dense<[2, 3, 1, 4, 1, 2, 3, 1, 4, 1, 2, 3, 3, 4, 1, 2]> :"
            " tensor<16xi64>",
            txt,
        )
        self.assertIn(
            "transition_matrix = dense<[5, 1, 2, 1, 1, 6, 1, 2, 2, 1, 7, 1, 1, 2,"
            " 1, 8]> : tensor<16xi64>",
            txt,
        )
        self.assertIn(
            "partial_dot = dense<[9, 2, 3, 5, 7, 1, 4, 6]> : tensor<8xi64>", txt
        )
        self.assertIn("partial_col = dense<[2, 3, 5, 1, 4, 2]> : tensor<6xi64>", txt)

    def test_reference_body_byte_matches(self) -> None:
        # The composite's decomposition (`_permute_from_operands`, int-literal
        # matrices + operand constants) is the semantics the emitter must match, so
        # it must byte-match the same pure-Python reference the generic body does.
        # Called directly (eager) — the marker itself can't be compiled until the
        # emitter ships.
        perm = _dedicated_perm()
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _P, size=_WIDTH, dtype=np.int64)
            operands = sparse_mod._abi_operands(perm, _to_field(canon))
            out = sparse_mod._permute_from_operands(perm, *operands)
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon])
            self.assertEqual(got, want)

    def test_fused_region_spec_is_dedicated(self) -> None:
        # On the dedicated path the ABI spec is live (not the inert stub): 6
        # operands, a callable body, and attrs carrying the permutation
        # discriminator plus the four matrices for a region wrapper (e.g. a Merkle
        # commit) to route the whole region through SparsePoseidonFusion.
        perm = _dedicated_perm()
        operands, body, attrs = perm.fused_region_spec(fnp.arange(_WIDTH, dtype=F))
        self.assertLen(operands, 6)
        self.assertTrue(callable(body))
        self.assertEqual(attrs["permutation"], "sparse_poseidon")
        for key in ("mds", "transition_matrix", "partial_dot", "partial_col"):
            self.assertIn(key, attrs)


# --- Self-contained equivalence: derive the sparse factorization from a RANDOM
#     naive instance and byte-match SparsePoseidon against an independent naive
#     schedule. Unlike `_reference_permute` (which transcribes the folded/sparse
#     schedule and so can only catch a lowering typo), this tests the optimization
#     itself: it runs a dense-MDS-every-round naive Poseidon, derives P / the
#     per-round sparse pairs / the folded constants via the standard M = Sp·Mp
#     recursion, and asserts the two agree. A wrong schedule fails here.
#
#     All derivation arithmetic is pure-Python int mod p — independent of the JAX
#     lowering under test.

_EQ_P = int(_P)


def _minv(a: int) -> int:
    return pow(a % _EQ_P, _EQ_P - 2, _EQ_P)


def _mm(A: list, B: list) -> list:
    n, k, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) % _EQ_P for j in range(m)]
        for i in range(n)
    ]


def _mv(A: list, v: list) -> list:
    return [sum(A[i][j] * v[j] for j in range(len(v))) % _EQ_P for i in range(len(A))]


def _matinv(A: list) -> list:
    n = len(A)
    aug = [
        [A[i][j] % _EQ_P for j in range(n)] + [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        piv = next(r for r in range(col, n) if aug[r][col] % _EQ_P != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        ipiv = _minv(aug[col][col])
        aug[col] = [(x * ipiv) % _EQ_P for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] % _EQ_P != 0:
                f = aug[r][col]
                aug[r] = [(aug[r][k] - f * aug[col][k]) % _EQ_P for k in range(2 * n)]
    return [row[n:] for row in aug]


def _blockdiag1(N: list) -> list:
    w = len(N) + 1
    B = [[0] * w for _ in range(w)]
    B[0][0] = 1
    for i in range(len(N)):
        for j in range(len(N)):
            B[i + 1][j + 1] = N[i][j]
    return B


def _rand_invertible(w: int, rng: np.random.Generator) -> list:
    while True:
        M = [[int(rng.integers(0, _EQ_P)) for _ in range(w)] for _ in range(w)]
        try:
            _matinv(M)
            _matinv([row[1:] for row in M[1:]])  # M-hat must be invertible too
            return M
        except StopIteration:
            continue


def _sbox(x: int) -> int:
    return pow(x % _EQ_P, _ALPHA, _EQ_P)


def _naive_permute(
    state: list, M: list, iarc: list, pre: list, trans_c: list, part_c: list, post: list
) -> list:
    """Naive Hades in SparsePoseidon's conventions: dense MDS EVERY round,
    full-width constants, S-box on lane 0 in partial rounds, `S-box -> ARC ->
    matrix` order, final round with no trailing constant. The transition round is
    just another dense-MDS full round here (the optimized form is what turns its
    matrix into P)."""
    w = _WIDTH
    s = [(state[i] + iarc[i]) % _EQ_P for i in range(w)]

    def full(vec: list, rc: list) -> list:
        return _mv(M, [(_sbox(vec[i]) + rc[i]) % _EQ_P for i in range(w)])

    for k in range(_HALF - 1):
        s = full(s, pre[k])
    s = full(s, trans_c)  # transition round (dense M in the naive form)
    for i in range(_NPART):
        s = _mv(
            M,
            [(_sbox(s[0]) + part_c[i][0]) % _EQ_P]
            + [(s[t] + part_c[i][t]) % _EQ_P for t in range(1, w)],
        )
    for k in range(_HALF - 1):
        s = full(s, post[k])
    return _mv(M, [_sbox(x) for x in s])  # final, no constant


def _derive_sparse(M: list, trans_c: list, part_c: list) -> tuple:
    """Factor the naive partial segment into (P, [Sp_i], [prc_i], folded trans).

    Sp_i = B_{i+1}^{-1} M B_i with B_i = blockdiag(1, M-hat^{-(NP-i)}) is the
    standard sparse factor; P = blockdiag(1, M-hat^{NP}) M is the leftover dense
    factor pushed into the transition round. The per-round full-width constants
    fold to one lane-0 scalar each, with the residual pushed back into the
    transition constant."""
    w, npr = _WIDTH, _NPART
    Mhat_inv = _matinv([row[1:] for row in M[1:]])
    N: list = [None] * (npr + 1)
    N[npr] = [[1 if i == j else 0 for j in range(w - 1)] for i in range(w - 1)]
    for i in range(npr - 1, -1, -1):
        N[i] = _mm(Mhat_inv, N[i + 1])
    B = [_blockdiag1(N[i]) for i in range(npr + 1)]
    Binv = [_matinv(B[i]) for i in range(npr + 1)]
    Sp = [_mm(_mm(Binv[i + 1], M), B[i]) for i in range(npr)]
    Pmat = _mm(Binv[0], M)
    Pinv = _matinv(Pmat)
    r = [_mv(_mm(Binv[i + 1], M), part_c[i]) for i in range(npr)]
    prc = [0] * npr
    carry = [0] * w
    for i in range(npr - 1, -1, -1):
        need = [(r[i][j] + carry[j]) % _EQ_P for j in range(w)]
        c0 = [Sp[i][t][0] for t in range(w)]  # Sp_i column 0
        row0tail = Sp[i][0][1:]
        rt_ct = sum(row0tail[t] * c0[t + 1] for t in range(w - 1)) % _EQ_P
        rt_needt = sum(row0tail[t] * need[t + 1] for t in range(w - 1)) % _EQ_P
        prc[i] = ((need[0] - rt_needt) * _minv((c0[0] - rt_ct) % _EQ_P)) % _EQ_P
        carry = [0] + [(need[t + 1] - prc[i] * c0[t + 1]) % _EQ_P for t in range(w - 1)]
    trans_folded = [(trans_c[j] + _mv(Pinv, carry)[j]) % _EQ_P for j in range(w)]
    return Pmat, Sp, prc, trans_folded


class SparsePoseidonEquivalenceTest(absltest.TestCase):
    def test_sparse_matches_naive_dense(self) -> None:
        rng = np.random.default_rng(20260725)
        w = _WIDTH

        def vec() -> list:
            return [int(rng.integers(0, _EQ_P)) for _ in range(w)]

        M = _rand_invertible(w, rng)
        iarc = vec()
        pre = [vec() for _ in range(_HALF - 1)]
        trans_c = vec()
        part_c = [vec() for _ in range(_NPART)]
        post = [vec() for _ in range(_HALF - 1)]

        Pmat, Sp, prc, trans_folded = _derive_sparse(M, trans_c, part_c)
        # Sp_i must come out sparse: rows >= 1 are e_t + (col scale)·e_0.
        for i in range(_NPART):
            for t in range(1, w):
                for j in range(1, w):
                    self.assertEqual(Sp[i][t][j] % _EQ_P, 1 if t == j else 0)

        params = SparsePoseidonParams(
            width=w,
            dtype=F,
            alpha=_ALPHA,
            half_full_rounds=_HALF,
            n_partial_rounds=_NPART,
            initial_arc=_fld(iarc),
            full_rc_pre=_fld(pre),
            transition_rc=_fld(trans_folded),
            partial_rc=_fld([prc[i] for i in range(_NPART)]),
            full_rc_post=_fld(post),
            mds=_fld(M),
            transition_matrix=_fld(Pmat),
            partial_dot=_fld([Sp[i][0] for i in range(_NPART)]),
            partial_col=_fld(
                [[Sp[i][t][0] for t in range(1, w)] for i in range(_NPART)]
            ),
        )
        perm = SparsePoseidon(params)
        rng_in = np.random.default_rng(7)
        for _ in range(8):
            canon = rng_in.integers(0, _EQ_P, size=w, dtype=np.int64)
            got = [int(x) for x in _to_canon(perm.permute(_to_field(canon)))]
            want = _naive_permute(
                [int(x) for x in canon], M, iarc, pre, trans_c, part_c, post
            )
            self.assertEqual(got, want)


if __name__ == "__main__":
    absltest.main()
