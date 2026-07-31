# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""DEEP-ALI composition against an independently divided quotient.

`deep_composition` is checked against the quotient built by explicit polynomial
division `(p(x) − p(ξ))/(x − ξ)` — a genuine polynomial exactly when the opening
is correct — so the test needs no interpolation or root convention, with base
and extension committed columns kept in their split blocks. `open_columns` is checked
by the barycentric round trip: Lagrange weights at `z` recover `p(z)`.
"""

from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.reed_solomon import eval_domain
from zorch.pcs.deep import (
    deep_composition,
    open_columns,
)
from zorch.poly.univariate import compute_lagrange_basis, eval_coeffs, powers
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.utils.field import split_coeffs

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont

_N_BITS = 4  # base domain N = 16
_B = 2  # base committed columns
_C = 3  # extension committed columns


def _quotient_coeffs(coeffs: Array, xi: Array) -> Array:
    """Coeffs of `(p(x) − p(ξ))/(x − ξ)` for `p` given by ascending `coeffs`.

    `p(x) − p(ξ) = Σ_k c_k (x^k − ξ^k) = (x − ξ)·Σ_j x^j Σ_{k>j} c_k ξ^{k−1−j}`,
    so the quotient's `x^j` coefficient is `Σ_{k>j} c_k ξ^{k−1−j}` — computed
    directly, no root convention involved."""
    n = coeffs.shape[0]
    xi_pows = powers(xi, n)  # ξ^0 .. ξ^{n-1}
    return fnp.stack(
        [fnp.sum(coeffs[j + 1 :] * xi_pows[: n - 1 - j]) for j in range(n - 1)]
    )


def _coeffs(x: Array) -> tuple[int, ...]:
    """An extension scalar as its base coefficients, for exact equality."""
    return tuple(int(v) for v in split_coeffs(x))


def _evals_on(coeffs: Array, domain: Array) -> Array:
    """Column of `p`'s evaluations over `domain`."""
    return fnp.stack([eval_coeffs(coeffs, x) for x in domain])


class DeepCompositionTest(absltest.TestCase):
    def _setup(self, seed: int) -> tuple[list[Array], Array, Array, Array, Array]:
        n = 1 << _N_BITS
        # Both blocks are degree < N — the low-degree premise DEEP rests on.
        base_coeffs = [rand_field(seed + i, (n,), KB) for i in range(_B)]
        ext_coeffs = [rand_ext_field(seed + 10 + j, (n,), KB, EF) for j in range(_C)]
        domain = eval_domain(KB, n)  # base subgroup — any distinct points work
        base_cols = fnp.stack([_evals_on(c, domain) for c in base_coeffs], axis=1)
        ext_cols = fnp.stack([_evals_on(c, domain) for c in ext_coeffs], axis=1)
        vf = rand_ext_field(seed + 100, (), KB, EF)
        return base_coeffs + ext_coeffs, domain, base_cols, ext_cols, vf

    def _expect(
        self, coeffs: list[Array], domain: Array, vf: Array, xis_of: list[Array]
    ) -> Array:
        """`Σ_m vf^m · q_m(x)` over `domain`, `q_m` the explicit quotient at its
        opening point `xis_of[m]`."""
        vf_pows = powers(vf, len(coeffs))
        want: Array | None = None
        for m, c in enumerate(coeffs):
            qm = _quotient_coeffs(c, xis_of[m])
            term = vf_pows[m] * _evals_on(qm, domain)
            want = term if want is None else want + term
        assert want is not None
        return want

    def test_batched_quotient_single_opening(self) -> None:
        coeffs, domain, base_cols, ext_cols, vf = self._setup(1)
        z = rand_ext_field(999, (), KB, EF)
        m = _B + _C
        evals = fnp.stack([eval_coeffs(coeffs[i], z) for i in range(m)])
        got = deep_composition(
            base_cols, ext_cols, evals, fnp.stack([z]), [0] * m, vf, domain
        )
        want = self._expect(coeffs, domain, vf, [z] * m)
        self.assertTrue(bool(fnp.all(got == want)), "batched quotient mismatch")

    def test_wrapped_opening_points(self) -> None:
        # Columns open at two distinct points; opening_pos maps each column.
        coeffs, domain, base_cols, ext_cols, vf = self._setup(7)
        m = _B + _C
        xis = rand_ext_field(555, (2,), KB, EF)
        pos = [i % 2 for i in range(m)]
        xis_of = [xis[pos[i]] for i in range(m)]
        evals = fnp.stack([eval_coeffs(coeffs[i], xis_of[i]) for i in range(m)])
        got = deep_composition(base_cols, ext_cols, evals, xis, pos, vf, domain)
        want = self._expect(coeffs, domain, vf, xis_of)
        self.assertTrue(bool(fnp.all(got == want)), "wrapped-opening mismatch")


class DeepCompositionPowerAssignmentTest(absltest.TestCase):
    def test_custom_vf_pows_reverses_assignment(self) -> None:
        # A caller may fix descending powers (Horner-style accumulation:
        # column 0 highest). Σ_j vf^(m−1−j)·q_j equals the ascending reference
        # over the REVERSED column list — exact, so byte equality.
        coeffs, domain, base_cols, ext_cols, vf = DeepCompositionTest._setup(self, 11)
        m = _B + _C
        z = rand_ext_field(888, (), KB, EF)
        evals = fnp.stack([eval_coeffs(coeffs[i], z) for i in range(m)])
        got = deep_composition(
            base_cols,
            ext_cols,
            evals,
            fnp.stack([z]),
            [0] * m,
            vf,
            domain,
            vf_pows=powers(vf, m)[::-1],
        )
        want = DeepCompositionTest._expect(
            self, list(reversed(coeffs)), domain, vf, [z] * m
        )
        self.assertTrue(bool(fnp.all(got == want)), "descending vf_pows mismatch")


class OpenColumnsTest(absltest.TestCase):
    def test_matches_per_column_dot(self) -> None:
        # The coefficient-wise reduction against its definition: each opening
        # is the plain Σ_k weights[k, pos_m]·col_m[k] extension dot, with mixed
        # opening runs, a subsampling stride, and blocks wide enough to split
        # into runs both ways.
        n, b, c, stride = 1 << _N_BITS, 5, 4, 2
        base_cols = fnp.stack(
            [rand_field(20 + i, (n * stride,), KB) for i in range(b)], axis=1
        )
        ext_cols = fnp.stack(
            [rand_ext_field(40 + j, (n * stride,), KB, EF) for j in range(c)],
            axis=1,
        )
        weights = fnp.stack(
            [rand_ext_field(60 + k, (n,), KB, EF) for k in range(2)], axis=1
        )
        pos = [0, 0, 1, 1, 1, 1, 0, 0, 1]  # base runs 0,1; ext runs 1,0,1
        got = open_columns(base_cols, ext_cols, weights, pos, stride=stride)
        for m in range(b + c):
            column = base_cols[::stride, m] if m < b else ext_cols[::stride, m - b]
            want = fnp.sum(weights[:, pos[m]] * column)
            self.assertEqual(_coeffs(got[m]), _coeffs(want), f"column {m}")

    def test_single_field_block(self) -> None:
        # One block empty (all columns cubic, or all base): the empty side's
        # column-index gather must stay integer-typed, and the opening still
        # recovers the direct dot. The wired DEEP opens an all-cubic quotient
        # column with no base block, so this is a real shape, not a corner.
        n, stride = 1 << _N_BITS, 2
        weights = rand_ext_field(7, (n,), KB, EF)[:, None]  # (N, 1)

        def _base_block(count: int) -> Array:
            if count == 0:
                return fnp.zeros((n * stride, 0), KB)
            return fnp.stack(
                [rand_field(80 + i, (n * stride,), KB) for i in range(count)], axis=1
            )

        def _ext_block(count: int) -> Array:
            if count == 0:
                return fnp.zeros((n * stride, 0), EF)
            return fnp.stack(
                [rand_ext_field(90 + i, (n * stride,), KB, EF) for i in range(count)],
                axis=1,
            )

        for base_c, ext_c in ((0, 3), (3, 0)):
            base_cols = _base_block(base_c)
            ext_cols = _ext_block(ext_c)
            m = base_c + ext_c
            got = open_columns(base_cols, ext_cols, weights, [0] * m, stride=stride)
            for col in range(m):
                column = (
                    base_cols[::stride, col]
                    if col < base_c
                    else ext_cols[::stride, col - base_c]
                )
                want = fnp.sum(weights[:, 0] * column)
                self.assertEqual(_coeffs(got[col]), _coeffs(want), f"column {col}")

    def test_recovers_direct_eval(self) -> None:
        # Barycentric round trip: Σ_i L_i(z)·p(domain[i]) == p(z), one base + one
        # extension column, split as the composition consumes them.
        n = 1 << _N_BITS
        domain = eval_domain(KB, n)
        z = rand_ext_field(42, (), KB, EF)
        weights = compute_lagrange_basis(z, domain.astype(EF))[:, None]  # (N, 1)
        base_coeffs = rand_field(3, (n,), KB)
        ext_coeffs = rand_ext_field(4, (n,), KB, EF)
        base_col = _evals_on(base_coeffs, domain)[:, None]  # (N, 1) base
        ext_col = _evals_on(ext_coeffs, domain)[:, None]  # (N, 1) extension
        got = open_columns(base_col, ext_col, weights, [0, 0], stride=1)
        self.assertEqual(_coeffs(got[0]), _coeffs(eval_coeffs(base_coeffs, z)))
        self.assertEqual(_coeffs(got[1]), _coeffs(eval_coeffs(ext_coeffs, z)))


if __name__ == "__main__":
    absltest.main()
