# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.poly.eq import (
    _OUTER_SPLIT_MIN,
    contract_hypercube_step,
    eq_factor,
    eq_root,
    eval_eq,
    expand_eq_family,
    expand_eq_to_hypercube,
    expand_hypercube_step,
    expand_monomial_step,
    expand_monomial_to_hypercube,
)

KB = zk_dtypes.koalabear_mont


class ExpandEqTest(absltest.TestCase):
    def test_partition_of_unity(self) -> None:
        # Σ_w eq(w, x) == 1 for any x.
        x = fnp.array([2, 5, 7], dtype=KB)
        out = expand_eq_to_hypercube(x, fnp.ones([], dtype=KB))
        self.assertEqual(out.shape, (8,))
        self.assertEqual(int(out.sum()), 1)

    def test_msb_first_indexing(self) -> None:
        # result[nat(w)] with w[0] as MSB: x=[2,3] -> result[2]=eq([1,0],x)=2*(1-3).
        x = fnp.array([2, 3], dtype=KB)
        out = expand_eq_to_hypercube(x, fnp.ones([], dtype=KB))
        # 2*(1-3) = 2*(-2) = -4 ≡ p-4 in KB; compare via field arithmetic.
        expected = fnp.array(2, dtype=KB) * (
            fnp.array(1, dtype=KB) - fnp.array(3, dtype=KB)
        )
        self.assertEqual(int(out[2]), int(expected))

    def test_msb_true_threads_to_step(self) -> None:
        # msb=True concatenates [low, high] each layer (x[j] at bit j) — it must
        # match the by-hand `expand_hypercube_step(msb=True)` doubling.
        x = fnp.array([2, 5, 7], dtype=KB)
        one = fnp.ones([], dtype=KB)
        out = expand_eq_to_hypercube(x, one, msb=True)
        ref = fnp.atleast_1d(one)
        for j in range(x.shape[0]):
            ref = expand_hypercube_step(ref, x[j], msb=True)
        self.assertTrue(bool(fnp.all(out == ref)))
        self.assertEqual(int(out.sum()), 1)  # still a partition of unity

    def test_outer_split_matches_doubling_chain(self) -> None:
        # At _OUTER_SPLIT_MIN variables the table is built as an outer product
        # of two half tables; it must stay exactly equal to the pure doubling
        # chain for both index conventions. The chain reference runs eagerly, so
        # the case list is kept minimal: the split/seed logic is per-call, and
        # one extension-field case covers the ones()-seeded second half (the
        # software GF(2^128) multiply is what makes a wider sweep time out on
        # the small-size CI budget).
        cases = [(KB, False), (KB, True), (zk_dtypes.binary_field_ghash, True)]
        for dtype, msb in cases:
            x = fnp.array(list(range(1, 17)), dtype=dtype)
            scalar = fnp.array(3, dtype=dtype)
            out = expand_eq_to_hypercube(x, scalar, msb=msb)
            ref = fnp.atleast_1d(scalar)
            for j in range(x.shape[0]):
                ref = expand_hypercube_step(ref, x[j], msb=msb)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(bool(fnp.all(out == ref)), f"dtype={dtype} msb={msb}")


class ExpandMonomialTest(absltest.TestCase):
    def test_msb_first_indexing(self) -> None:
        # result[nat(w)] with w[0] as MSB binding x[0]: x=[2,3] ->
        # result[2] = x[0] (bit 1 set, bit 0 clear).
        x = fnp.array([2, 3], dtype=KB)
        out = expand_monomial_to_hypercube(x, fnp.ones([], dtype=KB))
        self.assertEqual(out.shape, (4,))
        self.assertEqual([int(v) for v in out], [1, 3, 2, 6])

    def test_scalar_seeds_every_entry(self) -> None:
        x = fnp.array([2, 3], dtype=KB)
        out = expand_monomial_to_hypercube(x, fnp.array(5, dtype=KB))
        self.assertEqual([int(v) for v in out], [5, 15, 10, 30])

    def test_outer_split_matches_doubling_chain(self) -> None:
        # The monomial expand splits at _OUTER_SPLIT_MIN exactly as its eq twin
        # does, and must stay byte-equal to the pure doubling chain: a monomial
        # entry factors over any coordinate split (the first k coordinates own
        # the high index bits) and GF multiplication is associative and exact.
        # Case list kept minimal for the same reason as the eq twin's — the
        # eager chain reference over a software GF(2^128) multiply is what makes
        # a wider sweep time out on the small-size CI budget.
        for dtype in (KB, zk_dtypes.binary_field_ghash):
            n = _OUTER_SPLIT_MIN
            x = fnp.array(list(range(1, n + 1)), dtype=dtype)
            scalar = fnp.array(3, dtype=dtype)
            out = expand_monomial_to_hypercube(x, scalar)
            ref = fnp.atleast_1d(scalar)
            for j in range(n):
                ref = expand_monomial_step(ref, x[j])
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(bool(fnp.all(out == ref)), f"dtype={dtype}")

    def test_odd_variable_count_splits_unevenly(self) -> None:
        # n//2 leaves the halves unequal at odd n; the high/low bit assignment
        # must still line up with the chain.
        n = _OUTER_SPLIT_MIN + 1
        x = fnp.array(list(range(1, n + 1)), dtype=KB)
        scalar = fnp.array(7, dtype=KB)
        out = expand_monomial_to_hypercube(x, scalar)
        ref = fnp.atleast_1d(scalar)
        for j in range(n):
            ref = expand_monomial_step(ref, x[j])
        self.assertEqual(out.shape, (1 << n,))
        self.assertTrue(bool(fnp.all(out == ref)))


def _chain_family(cs: Array, *, msb: bool, suffix: bool) -> list[Array]:
    """Reference doubling chain over `cs`, retaining every layer."""
    order = range(cs.shape[0] - 1, -1, -1) if suffix else range(cs.shape[0])
    state = fnp.ones(1, dtype=cs.dtype)
    tables = []
    for j in order:
        state = expand_hypercube_step(state, cs[j], msb=msb)
        tables.append(state)
    return tables


class ExpandEqFamilyTest(absltest.TestCase):
    def test_small_families_match_the_step_loops(self) -> None:
        # Each member is re-derived from its slice alone so the contract does
        # not lean on chain nesting.
        cs = fnp.array([2, 5, 7], dtype=KB)
        for msb in (False, True):
            for suffix in (False, True):
                fam = expand_eq_family(cs, msb=msb, suffix=suffix)
                self.assertEqual([t.shape for t in fam], [(2,), (4,), (8,)])
                for i, table in enumerate(fam):
                    s = cs[3 - 1 - i :] if suffix else cs[: i + 1]
                    ref = _chain_family(s, msb=msb, suffix=suffix)[-1]
                    self.assertTrue(
                        bool(fnp.all(table == ref)), f"msb={msb} suffix={suffix} i={i}"
                    )

    def test_outer_split_family_matches_doubling_chain(self) -> None:
        # At _OUTER_SPLIT_MIN variables every member past the split point is
        # emitted as an outer product of the shared half; the whole family must
        # stay exactly equal to the retained doubling chain. KB carries the
        # four direction/placement combinations and one extension-field case
        # covers the compute_eq_evaluations combo — the eager GF(2^128) chain
        # reference is the CI cost that keeps the case list minimal (see the
        # expand_eq_to_hypercube split test above).
        cases = [
            (KB, False, False),
            (KB, False, True),
            (KB, True, False),
            (KB, True, True),
            (zk_dtypes.binary_field_ghash, True, True),
        ]
        for dtype, msb, suffix in cases:
            cs = fnp.array(list(range(1, 17)), dtype=dtype)
            fam = expand_eq_family(cs, msb=msb, suffix=suffix)
            refs = _chain_family(cs, msb=msb, suffix=suffix)
            for i, (table, ref) in enumerate(zip(fam, refs, strict=True)):
                self.assertEqual(table.shape, ref.shape)
                self.assertTrue(
                    bool(fnp.all(table == ref)),
                    f"dtype={dtype} msb={msb} suffix={suffix} i={i}",
                )

    def test_keep_masks_and_skips_dead_members(self) -> None:
        # The keep contract is predicate-defined, not split-regime-defined:
        # every kept member stays byte-equal to the full build (whose
        # arithmetic the split test above pins per-dtype — keep is list
        # plumbing on top of the same emissions, hence KB only), every
        # un-kept slot is None. Two predicate shapes:
        #   * n straddling the split, odd indices: chains masked below,
        #     product emissions skipped above;
        #   * n at the split, one kept product: its shared base factor
        #     (index n//2 - 1) is un-kept publicly yet must still be built
        #     as a building block — in BOTH orientations (the odd case hits
        #     this only for suffix families, whose base is the back half).
        cases = [
            (_OUTER_SPLIT_MIN + 1, lambda i: i % 2 == 1),
            (_OUTER_SPLIT_MIN, lambda i: i == _OUTER_SPLIT_MIN // 2),
        ]
        for n, keep in cases:
            cs = fnp.array(list(range(1, n + 1)), dtype=KB)
            for msb in (False, True):
                for suffix in (False, True):
                    full = expand_eq_family(cs, msb=msb, suffix=suffix)
                    kept = expand_eq_family(cs, msb=msb, suffix=suffix, keep=keep)
                    self.assertLen(kept, n)
                    for i, (table, ref) in enumerate(zip(kept, full, strict=True)):
                        ctx = f"n={n} msb={msb} suffix={suffix} i={i}"
                        if not keep(i):
                            self.assertIsNone(table, ctx)
                            continue
                        self.assertIsNotNone(table, ctx)
                        self.assertTrue(bool(fnp.all(table == ref)), ctx)


class EqFactorTest(absltest.TestCase):
    def test_matches_eval_eq_on_length_one_points(self) -> None:
        t = fnp.array(5, dtype=KB)
        z = fnp.array(9, dtype=KB)
        self.assertEqual(int(eq_factor(t, z)), int(eval_eq(t[None], z[None])))

    def test_selects_coordinate_at_boolean_t(self) -> None:
        # eq(0, z) = 1 - z and eq(1, z) = z: the factor a bound bit selects.
        z = fnp.array(7, dtype=KB)
        one = fnp.ones((), dtype=KB)
        self.assertEqual(int(eq_factor(fnp.zeros((), KB), z)), int(one - z))
        self.assertEqual(int(eq_factor(one, z)), int(z))

    def test_broadcasts_elementwise(self) -> None:
        t = fnp.array([0, 1, 3], dtype=KB)
        z = fnp.array([5, 5, 5], dtype=KB)
        out = eq_factor(t, z)
        self.assertEqual(out.shape, (3,))
        for i in range(3):
            self.assertEqual(int(out[i]), int(eq_factor(t[i], z[i])))


class EqRootTest(absltest.TestCase):
    def test_factor_vanishes_at_root(self) -> None:
        z = fnp.array(11, dtype=KB)
        self.assertEqual(int(eq_factor(eq_root(z), z)), 0)

    def test_broadcasts_over_points(self) -> None:
        z = fnp.array([3, 5, 7], dtype=KB)
        roots = eq_root(z)
        self.assertEqual(roots.shape, (3,))
        self.assertTrue(bool(fnp.all(eq_factor(roots, z) == fnp.zeros((3,), KB))))


class ContractHypercubeStepTest(absltest.TestCase):
    def test_sums_adjacent_pairs(self) -> None:
        state = fnp.array([1, 2, 3, 4, 5, 6], dtype=KB)
        out = contract_hypercube_step(state)
        self.assertTrue(bool(fnp.all(out == fnp.array([3, 7, 11], dtype=KB))))

    def test_marginalizes_the_lsb_variable(self) -> None:
        # Σ_b eq((w, b), x) = eq(w, x[:-1]): contracting the expanded table for
        # x recovers the table for x without its LSB coordinate, exactly.
        x = fnp.array([2, 5, 7], dtype=KB)
        one = fnp.ones((), dtype=KB)
        full = expand_eq_to_hypercube(x, one)
        want = expand_eq_to_hypercube(x[:-1], one)
        self.assertTrue(bool(fnp.all(contract_hypercube_step(full) == want)))

    def test_inverts_expand_step(self) -> None:
        # expand splits each entry into (1-coord)/coord shares; contracting
        # sums them back to the entry exactly.
        state = fnp.array([3, 9, 4, 6], dtype=KB)
        expanded = expand_hypercube_step(state, fnp.array(5, KB))
        self.assertTrue(bool(fnp.all(contract_hypercube_step(expanded) == state)))


class EvalEqTest(absltest.TestCase):
    def test_matches_hypercube_inner_product(self) -> None:
        # eq is the reproducing kernel: eq(w,x) == Σ_v eq(v,w)·eq(v,x).
        w = fnp.array([3, 5, 7], dtype=KB)
        x = fnp.array([2, 9, 4], dtype=KB)
        ew = expand_eq_to_hypercube(w, fnp.ones([], dtype=KB))
        ex = expand_eq_to_hypercube(x, fnp.ones([], dtype=KB))
        self.assertEqual(int(eval_eq(w, x)), int((ew * ex).sum()))

    def test_one_on_equal_boolean_points(self) -> None:
        a = fnp.array([1, 0, 1], dtype=KB)
        self.assertEqual(int(eval_eq(a, a)), 1)
        b = fnp.array([1, 1, 1], dtype=KB)
        self.assertEqual(int(eval_eq(a, b)), 0)


if __name__ == "__main__":
    absltest.main()
