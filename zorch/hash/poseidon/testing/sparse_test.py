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

import dataclasses
from collections.abc import Sequence
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import (
    babybear_mont,
    goldilocks_mont,  # wider than an int64, for the gate test
    koalabear_mont,  # a distinct field, for dtype-guard tests
    pfinfo,
)

from zorch.fusion import FUSED_REGION_MARKER
from zorch.hash.permutation import Permutation
from zorch.hash.poseidon import sparse as sparse_mod
from zorch.hash.poseidon.params import SparsePoseidonParams
from zorch.hash.poseidon.sparse import (
    POSEIDON_SPARSE_MARKER,
    POSEIDON_SPARSE_MARKER_VERSION,
    SparsePoseidon,
)

# The field prime the canonical-int reference reduces mod.
_BABYBEAR_P = pfinfo(babybear_mont).modulus
# 2^64 - 2^32 + 1 — past what an int64 marker attribute holds.
_GOLDILOCKS_P = pfinfo(goldilocks_mont).modulus

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
    return fnp.asarray(canon.astype(np.int64).astype(babybear_mont))


def _to_canon(arr: fnp.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(arr).astype(object), dtype=object)


def _field(rows: object, dtype: Any) -> fnp.ndarray:
    """Canonical ints -> a field array. uint64, not int64: a canonical Goldilocks
    value can exceed an int64."""
    return fnp.array(np.array(rows, dtype=np.uint64), dtype=dtype)


def _fld(rows: object) -> fnp.ndarray:
    return _field(rows, babybear_mont)


@dataclasses.dataclass(frozen=True)
class _Schedule:
    """One optimized-sparse Poseidon parameterization, as canonical ints.

    Both sides of a byte-match read the same instance: `params` renders it for
    zorch, `_reference_permute` interprets it in pure Python, so the two cannot
    drift apart. Field names match `SparsePoseidonParams` so rendering needs no
    translation step.
    """

    p: int
    width: int
    alpha: int
    half_full_rounds: int
    n_partial_rounds: int
    initial_arc: Sequence[int]
    full_rc_pre: Sequence[Sequence[int]]
    transition_rc: Sequence[int]
    partial_rc: Sequence[int]
    full_rc_post: Sequence[Sequence[int]]
    mds: Sequence[Sequence[int]]
    transition_matrix: Sequence[Sequence[int]]
    partial_dot: Sequence[Sequence[int]]
    partial_col: Sequence[Sequence[int]]

    def params(self, dtype: Any) -> SparsePoseidonParams:
        """Render for zorch. Kwargs are written out rather than splatted so a
        renamed field fails the type check, not the run."""
        return SparsePoseidonParams(
            width=self.width,
            dtype=dtype,
            alpha=self.alpha,
            half_full_rounds=self.half_full_rounds,
            n_partial_rounds=self.n_partial_rounds,
            initial_arc=_field(self.initial_arc, dtype),
            full_rc_pre=_field(self.full_rc_pre, dtype),
            transition_rc=_field(self.transition_rc, dtype),
            partial_rc=_field(self.partial_rc, dtype),
            full_rc_post=_field(self.full_rc_post, dtype),
            mds=_field(self.mds, dtype),
            transition_matrix=_field(self.transition_matrix, dtype),
            partial_dot=_field(self.partial_dot, dtype),
            partial_col=_field(self.partial_col, dtype),
        )


_WIDTH4 = _Schedule(
    p=_BABYBEAR_P,
    width=_WIDTH,
    alpha=_ALPHA,
    half_full_rounds=_HALF,
    n_partial_rounds=_NPART,
    initial_arc=_INITIAL_ARC,
    full_rc_pre=_FULL_RC_PRE,
    transition_rc=_TRANSITION_RC,
    partial_rc=_PARTIAL_RC,
    full_rc_post=_FULL_RC_POST,
    mds=_MDS,
    transition_matrix=_TRANSITION_M,
    partial_dot=_PARTIAL_DOT,
    partial_col=_PARTIAL_COL,
)


def _params() -> SparsePoseidonParams:
    return _WIDTH4.params(babybear_mont)


def _wide_field_params() -> SparsePoseidonParams:
    """The same width-4 schedule over Goldilocks (`p = 2^64 - 2^32 + 1`), with one
    MDS entry above `2^63 - 1`. alpha=7 is coprime to `p - 1` here too, so the
    S-box still permutes. Only the matrices matter to the marker gate — the round
    constants ride as operands whatever their magnitude — so those stay small."""
    mds = ((_GOLDILOCKS_P - 1, 3, 1, 4), (1, 2, 3, 1), (4, 1, 2, 3), (3, 4, 1, 2))
    wide = dataclasses.replace(_WIDTH4, p=_GOLDILOCKS_P, mds=mds)
    return wide.params(goldilocks_mont)


def _reference_permute(state_canon: list[int], sched: _Schedule) -> list[int]:
    """Independent optimized-sparse Poseidon reference in pure-Python int mod p.

    Mirrors the documented schedule: initial ARC, (half-1) full rounds with the
    dense MDS, one transition round with `P`, n_partial sparse rounds (S-box on
    lane 0, then lane-0 dot + lane-t rank-1 update), (half-1) full rounds, and a
    final full round with no trailing constant. The S-box precedes the round
    constant (the constant seeds the next round's S-box); the initial ARC seeds
    the first.
    """
    # Only what the closures below capture; everything else reads `sched.<field>`
    # so it stays diffable against SparsePoseidonParams.
    p, w, alpha = sched.p, sched.width, sched.alpha
    mds = sched.mds
    s = [x % p for x in state_canon]

    def sbox(x: int) -> int:
        return pow(x % p, alpha, p)

    def matmul(mat: Sequence[Sequence[int]], vec: list[int]) -> list[int]:
        return [sum(mat[i][j] * vec[j] for j in range(w)) % p for i in range(w)]

    def full_round(
        vec: list[int], rc: Sequence[int], mat: Sequence[Sequence[int]]
    ) -> list[int]:
        return matmul(mat, [(sbox(vec[i]) + rc[i]) % p for i in range(w)])

    s = [(s[i] + sched.initial_arc[i]) % p for i in range(w)]
    for r in range(sched.half_full_rounds - 1):
        s = full_round(s, sched.full_rc_pre[r], mds)
    s = full_round(s, sched.transition_rc, sched.transition_matrix)
    for r in range(sched.n_partial_rounds):
        a = (sbox(s[0]) + sched.partial_rc[r]) % p
        old = [a] + s[1:]
        new0 = sum(old[j] * sched.partial_dot[r][j] for j in range(w)) % p
        rest = [(s[t] + a * sched.partial_col[r][t - 1]) % p for t in range(1, w)]
        s = [new0] + rest
    for r in range(sched.half_full_rounds - 1):
        s = full_round(s, sched.full_rc_post[r], mds)
    s = matmul(mds, [sbox(x) for x in s])
    return s


class SparsePoseidonReferenceByteMatchTest(absltest.TestCase):
    def test_byte_matches_reference(self) -> None:
        perm = SparsePoseidon(_params())
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _BABYBEAR_P, size=_WIDTH, dtype=np.int64)
            out = perm.permute(_to_field(canon))
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon], _WIDTH4)
            self.assertEqual(got, want)


def _big_schedule() -> _Schedule:
    """A deterministic production-scale Goldilocks schedule: 8 lanes, 22 partial
    rounds.

    The width-4 config above carries only 2 partial rounds, which is why it could
    not catch a per-lane fan-out in the sparse partial layer — the cost of
    re-deriving each round per consumer compounds with the partial-round count,
    so 2 rounds hides what 22 makes fatal.

    The constants are arbitrary — this pins zorch against the independent
    reference, not against any particular parameterization — but they are
    full-magnitude so nothing folds away at trace time.
    """
    width, half, npart = 8, 4, 22
    rng = np.random.default_rng(565)

    def rnd(*shape: int) -> list:
        return rng.integers(1, _GOLDILOCKS_P, size=shape, dtype=np.uint64).tolist()

    return _Schedule(
        p=_GOLDILOCKS_P,
        width=width,
        alpha=_ALPHA,
        half_full_rounds=half,
        n_partial_rounds=npart,
        initial_arc=rnd(width),
        full_rc_pre=rnd(half - 1, width),
        transition_rc=rnd(width),
        partial_rc=rnd(npart),
        full_rc_post=rnd(half - 1, width),
        mds=rnd(width, width),
        transition_matrix=rnd(width, width),
        partial_dot=rnd(npart, width),
        partial_col=rnd(npart, width - 1),
    )


class SparsePoseidonManyPartialRoundsTest(absltest.TestCase):
    """The consequence of breaking the chained-input rule, at production scale.

    Chained over 22 partial rounds the per-lane form re-derives each round per
    consumer, and this test stops terminating. See `apply_sparse_partial` and
    https://github.com/fractalyze/zorch/issues/565.
    """

    @absltest.skipIf(
        frx.default_backend() == "gpu",
        "quarantined: frx.jit miscompiles the goldilocks square-of-add"
        " (u+v)*(u+v) on cuda — every full round's power(s + rc, alpha)"
        " contains it, so the whole permutation is wrong on the gpu backend"
        " regardless of schedule size; the cpu run keeps validating this"
        " byte-match. Tracked on the fractalyze xla work board:"
        " 'fix(gpu/codegen): jitted goldilocks mul(add,add) miscompiles on"
        " cuda'. Remove this skip with that fix's frx pin bump.",
    )
    def test_byte_matches_reference_at_production_scale(self) -> None:
        sched = _big_schedule()
        perm = SparsePoseidon(sched.params(goldilocks_mont))

        rng = np.random.default_rng(1)
        canon = rng.integers(1, _GOLDILOCKS_P, size=sched.width, dtype=np.uint64)
        state = _field(canon, goldilocks_mont)
        got = [int(x) for x in _to_canon(perm.permute(state))]
        want = _reference_permute([int(x) for x in canon], sched)
        self.assertEqual(got, want)


class SparsePoseidonPermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        perm = SparsePoseidon(_params())
        self.assertIsInstance(perm, Permutation)
        self.assertEqual(perm.width, _WIDTH)
        self.assertEqual(perm.dtype, babybear_mont)
        # The shipped emitter routes the permute to the dedicated sparse marker.
        self.assertTrue(perm.has_dedicated_fusion)

    def test_permute_shape_and_vmap(self) -> None:
        perm = SparsePoseidon(_params())
        x = fnp.arange(_WIDTH, dtype=babybear_mont)
        out = perm.permute(x)
        self.assertEqual(out.shape, (_WIDTH,))
        self.assertEqual(out.dtype, babybear_mont)
        batch = fnp.stack([x, x + babybear_mont(1)])
        bout = frx.vmap(perm.permute)(batch)
        self.assertEqual(bout.shape, (2, _WIDTH))
        self.assertTrue(bool(fnp.array_equal(bout[0], out)))

    def test_permute_rejects_wrong_shape(self) -> None:
        perm = SparsePoseidon(_params())
        with self.assertRaises(ValueError):
            perm.permute(fnp.zeros((_WIDTH + 1,), dtype=babybear_mont))
        with self.assertRaises(ValueError):
            perm.permute(fnp.zeros((2, _WIDTH), dtype=babybear_mont))

    def test_permute_rejects_wrong_dtype(self) -> None:
        # A right-shaped state in the wrong field must hit the TypeError branch,
        # not silently permute in the other field.
        perm = SparsePoseidon(_params())
        with self.assertRaises(TypeError):
            perm.permute(fnp.zeros((_WIDTH,), dtype=koalabear_mont))


class SparsePoseidonParamsValidationTest(absltest.TestCase):
    def test_value_equality_and_hash(self) -> None:
        self.assertEqual(_params(), _params())
        self.assertEqual(hash(_params()), hash(_params()))

    def test_rejects_wrong_partial_col_shape(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(
                _params(), partial_col=_fld(((2, 3, 5, 9), (1, 4, 2, 8)))
            )  # width, not width-1

    def test_rejects_dtype_mismatch(self) -> None:
        # An array in a different field than `dtype` must be rejected.
        wrong = fnp.asarray(
            np.array(_INITIAL_ARC, dtype=np.int64).astype(koalabear_mont)
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(_params(), initial_arc=wrong)

    def test_rejects_bad_scalars(self) -> None:
        # Each scalar guard fires independently.
        for field, bad in (
            ("alpha", 0),  # must be positive
            ("width", 1),  # must be >= 2 (else partial_col is (npr, 0))
            ("half_full_rounds", 0),  # must be positive
            ("n_partial_rounds", -1),  # must be non-negative
        ):
            with self.assertRaises(ValueError, msg=field):
                dataclasses.replace(_params(), **{field: bad})


class SparsePoseidonMarkerEmissionTest(absltest.TestCase):
    def test_matrix_wider_than_i64_falls_back_to_generic(self) -> None:
        # A matrix entry above 2^63 can only ride the attributes as a u64
        # bit-cast, and that wide range is gated on
        # `_WIDE_ATTR_EMITTER_AVAILABLE` (off until the pinned plugin carries
        # the sponge-hash sparse arm) — so the instance marks its region with
        # the generic "zorch.fused_region" name instead; the normal-form body
        # still fuses. Constructing it must not raise: the gate establishes
        # representability before `_rows_to_i64` builds an attribute, where an
        # out-of-range entry is an OverflowError. The dedicated (default) path
        # is covered by SparsePoseidonDedicatedMarkerTest.
        perm = SparsePoseidon(_wide_field_params())
        self.assertFalse(perm.has_dedicated_fusion)
        self.assertEqual(perm.fused_region_marker, (FUSED_REGION_MARKER, 0))
        txt = (
            frx.jit(perm.permute)
            .lower(fnp.arange(_WIDTH, dtype=goldilocks_mont))
            .as_text()
        )
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{FUSED_REGION_MARKER}"', composite_line)

    def test_wide_matrix_takes_dedicated_marker_when_emitter_available(
        self,
    ) -> None:
        # With the wide-attr gate flipped, the same Goldilocks instance selects
        # the dedicated marker, and the >= 2^63 entry encodes as the negative
        # i64 carrying identical bits (the emitter reinterprets as uint64).
        # Attribute-level only: compiling the marker needs the sponge-arm
        # plugin the gate stages.
        with mock.patch.object(sparse_mod, "_WIDE_ATTR_EMITTER_AVAILABLE", True):
            perm = SparsePoseidon(_wide_field_params())
        self.assertTrue(perm.has_dedicated_fusion)
        self.assertEqual(perm.fused_region_version, POSEIDON_SPARSE_MARKER_VERSION)
        attrs, version = sparse_mod._marker_attrs(perm)
        self.assertEqual(version, POSEIDON_SPARSE_MARKER_VERSION)
        mds = attrs["mds"]
        self.assertEqual(mds.dtype, np.int64)
        self.assertLess(int(mds[0]), 0)
        self.assertEqual(int(mds.view(np.uint64)[0]), _GOLDILOCKS_P - 1)

    def test_wide_dedicated_permute_lowers(self) -> None:
        # Lowering (not just attribute inspection): the wide ABI reference body
        # must TRACE — a bare Python-int literal past the staging cap raises
        # OverflowError at trace time, which attribute-level checks never see
        # (caught live on the first Goldilocks dedicated permute). The marker
        # and its 6-operand ABI must survive `_scale`'s in-field constant
        # assembly (nothing lifted).
        with mock.patch.object(sparse_mod, "_WIDE_ATTR_EMITTER_AVAILABLE", True):
            perm = SparsePoseidon(_wide_field_params())
            txt = (
                frx.jit(perm.permute)
                .lower(fnp.arange(_WIDTH).astype(goldilocks_mont))
                .as_text()
            )
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON_SPARSE_MARKER}"', composite_line)
        operands = composite_line.split(f'"{POSEIDON_SPARSE_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 6, composite_line)

    def test_rows_to_i64_bitcasts_wide_values(self) -> None:
        # The encoding is identity below 2^63 and a bit-cast above: round-trip
        # through uint64 recovers every canonical value exactly.
        rows = ((2**63, _GOLDILOCKS_P - 1), (1, 0))
        enc = sparse_mod._rows_to_i64(rows)
        self.assertEqual(enc.dtype, np.int64)
        self.assertEqual(
            [int(v) for v in enc.view(np.uint64)],
            [2**63, _GOLDILOCKS_P - 1, 1, 0],
        )


class SparsePoseidonDedicatedMarkerTest(absltest.TestCase):
    """The dedicated path — the default now that the plugin ships the emitter.
    Verified by lowering (marker name, attributes, ABI operand count) and by
    exercising the reference decomposition body directly. The end-to-end
    compile+run byte-match lives in `SparsePoseidonReferenceByteMatchTest`."""

    def test_permute_emits_dedicated_composite(self) -> None:
        # When the emitter is available the permute marks its region
        # "zorch.sparse_poseidon" so XLA routes it to SparsePoseidonFusion. The
        # ABI operands are exactly [state, initial_arc, full_rc_pre, transition_rc,
        # partial_rc, full_rc_post] = 6; a closed-over matrix would be lifted to a
        # leading operand (frx.lax.composite prepends consts) and break that ABI.
        perm = SparsePoseidon(_params())
        self.assertTrue(perm.has_dedicated_fusion)
        txt = (
            frx.jit(perm.permute)
            .lower(fnp.arange(_WIDTH, dtype=babybear_mont))
            .as_text()
        )
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
        perm = SparsePoseidon(_params())
        txt = (
            frx.jit(perm.permute)
            .lower(fnp.arange(_WIDTH, dtype=babybear_mont))
            .as_text()
        )
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
        # Exercised directly (eager) so this pins the reference body independently of
        # the compiled marker (whose end-to-end match lives in the byte-match test).
        perm = SparsePoseidon(_params())
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _BABYBEAR_P, size=_WIDTH, dtype=np.int64)
            operands = sparse_mod._abi_operands(perm, _to_field(canon))
            out = sparse_mod._permute_from_operands(perm, *operands)
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon], _WIDTH4)
            self.assertEqual(got, want)

    def test_fused_region_spec_is_dedicated(self) -> None:
        # On the dedicated path the ABI spec is live (not the inert stub): 6
        # operands, a callable body, and attrs carrying the permutation
        # discriminator plus the four matrices for a region wrapper (e.g. a Merkle
        # commit) to route the whole region through SparsePoseidonFusion.
        perm = SparsePoseidon(_params())
        operands, body, attrs = perm.fused_region_spec(
            fnp.arange(_WIDTH, dtype=babybear_mont)
        )
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


def _minv(a: int) -> int:
    return pow(a % _BABYBEAR_P, _BABYBEAR_P - 2, _BABYBEAR_P)


def _mm(A: list, B: list) -> list:
    n, k, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) % _BABYBEAR_P for j in range(m)]
        for i in range(n)
    ]


def _mv(A: list, v: list) -> list:
    return [
        sum(A[i][j] * v[j] for j in range(len(v))) % _BABYBEAR_P for i in range(len(A))
    ]


def _matinv(A: list) -> list:
    n = len(A)
    aug = [
        [A[i][j] % _BABYBEAR_P for j in range(n)]
        + [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        piv = next(r for r in range(col, n) if aug[r][col] % _BABYBEAR_P != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        ipiv = _minv(aug[col][col])
        aug[col] = [(x * ipiv) % _BABYBEAR_P for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] % _BABYBEAR_P != 0:
                f = aug[r][col]
                aug[r] = [
                    (aug[r][k] - f * aug[col][k]) % _BABYBEAR_P for k in range(2 * n)
                ]
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
        M = [[int(rng.integers(0, _BABYBEAR_P)) for _ in range(w)] for _ in range(w)]
        try:
            _matinv(M)
            _matinv([row[1:] for row in M[1:]])  # M-hat must be invertible too
            return M
        except StopIteration:
            continue


def _sbox(x: int) -> int:
    return pow(x % _BABYBEAR_P, _ALPHA, _BABYBEAR_P)


def _naive_permute(
    state: list, M: list, iarc: list, pre: list, trans_c: list, part_c: list, post: list
) -> list:
    """Naive Hades in SparsePoseidon's conventions: dense MDS EVERY round,
    full-width constants, S-box on lane 0 in partial rounds, `S-box -> ARC ->
    matrix` order, final round with no trailing constant. The transition round is
    just another dense-MDS full round here (the optimized form is what turns its
    matrix into P)."""
    w = _WIDTH
    s = [(state[i] + iarc[i]) % _BABYBEAR_P for i in range(w)]

    def full(vec: list, rc: list) -> list:
        return _mv(M, [(_sbox(vec[i]) + rc[i]) % _BABYBEAR_P for i in range(w)])

    for k in range(_HALF - 1):
        s = full(s, pre[k])
    s = full(s, trans_c)  # transition round (dense M in the naive form)
    for i in range(_NPART):
        s = _mv(
            M,
            [(_sbox(s[0]) + part_c[i][0]) % _BABYBEAR_P]
            + [(s[t] + part_c[i][t]) % _BABYBEAR_P for t in range(1, w)],
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
        need = [(r[i][j] + carry[j]) % _BABYBEAR_P for j in range(w)]
        c0 = [Sp[i][t][0] for t in range(w)]  # Sp_i column 0
        row0tail = Sp[i][0][1:]
        rt_ct = sum(row0tail[t] * c0[t + 1] for t in range(w - 1)) % _BABYBEAR_P
        rt_needt = sum(row0tail[t] * need[t + 1] for t in range(w - 1)) % _BABYBEAR_P
        prc[i] = (
            (need[0] - rt_needt) * _minv((c0[0] - rt_ct) % _BABYBEAR_P)
        ) % _BABYBEAR_P
        carry = [0] + [
            (need[t + 1] - prc[i] * c0[t + 1]) % _BABYBEAR_P for t in range(w - 1)
        ]
    trans_folded = [(trans_c[j] + _mv(Pinv, carry)[j]) % _BABYBEAR_P for j in range(w)]
    return Pmat, Sp, prc, trans_folded


class SparsePoseidonEquivalenceTest(absltest.TestCase):
    def test_sparse_matches_naive_dense(self) -> None:
        rng = np.random.default_rng(20260725)
        w = _WIDTH

        def vec() -> list:
            return [int(rng.integers(0, _BABYBEAR_P)) for _ in range(w)]

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
                    self.assertEqual(Sp[i][t][j] % _BABYBEAR_P, 1 if t == j else 0)

        params = SparsePoseidonParams(
            width=w,
            dtype=babybear_mont,
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
            canon = rng_in.integers(0, _BABYBEAR_P, size=w, dtype=np.int64)
            got = [int(x) for x in _to_canon(perm.permute(_to_field(canon)))]
            want = _naive_permute(
                [int(x) for x in canon], M, iarc, pre, trans_c, part_c, post
            )
            self.assertEqual(got, want)


if __name__ == "__main__":
    absltest.main()
