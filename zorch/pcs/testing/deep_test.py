# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""DEEP-ALI composition against an independently divided quotient.

`deep_composition` is checked against the quotient built by explicit polynomial
division `(p(x) − p(ξ))/(x − ξ)` — a genuine polynomial exactly when the opening
is correct — so the test needs no interpolation or root convention, with base
and cubic committed columns kept in their split blocks. `open_columns` is checked
by the barycentric round trip: Lagrange weights at `z` recover `p(z)`.
"""

from __future__ import annotations

import frx
import frx.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.reed_solomon import eval_domain
from zorch.pcs.deep import deep_composition, open_columns
from zorch.poly.univariate import compute_lagrange_basis, powers

KB = zk_dtypes.koalabear_mont
KB4 = zk_dtypes.koalabearx4_mont

_N_BITS = 4  # base domain N = 16
_B = 2  # base committed columns
_C = 3  # cubic committed columns


def _rand_base(n: int, seed: int) -> Array:
    """`n` random koalabear elements."""
    return jnp.array(
        np.random.default_rng(seed).integers(0, 1 << 30, n).astype(np.uint32), dtype=KB
    )


def _rand_ext(n: int, seed: int) -> Array:
    """`n` random koalabearx4 elements."""
    limbs = np.random.default_rng(seed).integers(0, 1 << 30, (n, 4)).astype(np.uint32)
    return frx.lax.bitcast_convert_type(jnp.array(limbs.astype(KB)), KB4).reshape(-1)


def _poly_eval(coeffs: Array, x: Array) -> Array:
    """`Σ_k coeffs[k]·x^k` for a scalar `x` (base or extension)."""
    return jnp.sum(coeffs * powers(x, coeffs.shape[0]))


def _quotient_coeffs(coeffs: Array, xi: Array) -> Array:
    """Coeffs of `(p(x) − p(ξ))/(x − ξ)` for `p` given by ascending `coeffs`.

    `p(x) − p(ξ) = Σ_k c_k (x^k − ξ^k) = (x − ξ)·Σ_j x^j Σ_{k>j} c_k ξ^{k−1−j}`,
    so the quotient's `x^j` coefficient is `Σ_{k>j} c_k ξ^{k−1−j}` — computed
    directly, no root convention involved."""
    n = coeffs.shape[0]
    xi_pows = powers(xi, n)  # ξ^0 .. ξ^{n-1}
    return jnp.stack(
        [jnp.sum(coeffs[j + 1 :] * xi_pows[: n - 1 - j]) for j in range(n - 1)]
    )


def _limbs(x: Array) -> tuple[int, ...]:
    """A koalabearx4 scalar as its base limbs, for exact equality."""
    return tuple(int(v) for v in frx.lax.bitcast_convert_type(x, KB).reshape(4))


def _evals_on(coeffs: Array, domain: Array) -> Array:
    """Column of `p`'s evaluations over `domain`."""
    return jnp.stack([_poly_eval(coeffs, x) for x in domain])


class DeepCompositionTest(absltest.TestCase):
    def _setup(self, seed: int) -> tuple[list[Array], Array, Array, Array, Array]:
        n = 1 << _N_BITS
        base_coeffs = [_rand_base(n, seed + i) for i in range(_B)]  # deg<N base
        cubic_coeffs = [_rand_ext(n, seed + 10 + j) for j in range(_C)]  # deg<N ext
        domain = eval_domain(KB, n)  # base subgroup — any distinct points work
        base_cols = jnp.stack([_evals_on(c, domain) for c in base_coeffs], axis=1)
        cubic_cols = jnp.stack([_evals_on(c, domain) for c in cubic_coeffs], axis=1)
        vf = _rand_ext(1, seed + 100)[0]
        return base_coeffs + cubic_coeffs, domain, base_cols, cubic_cols, vf

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
        coeffs, domain, base_cols, cubic_cols, vf = self._setup(1)
        z = _rand_ext(1, 999)[0]
        m = _B + _C
        evals = jnp.stack([_poly_eval(coeffs[i], z) for i in range(m)])
        got = deep_composition(
            base_cols, cubic_cols, evals, jnp.stack([z]), [0] * m, vf, domain
        )
        want = self._expect(coeffs, domain, vf, [z] * m)
        self.assertTrue(bool(jnp.all(got == want)), "batched quotient mismatch")

    def test_wrapped_opening_points(self) -> None:
        # Columns open at two distinct points; opening_pos maps each column.
        coeffs, domain, base_cols, cubic_cols, vf = self._setup(7)
        m = _B + _C
        xis = _rand_ext(2, 555)
        pos = [i % 2 for i in range(m)]
        xis_of = [xis[pos[i]] for i in range(m)]
        evals = jnp.stack([_poly_eval(coeffs[i], xis_of[i]) for i in range(m)])
        got = deep_composition(base_cols, cubic_cols, evals, xis, pos, vf, domain)
        want = self._expect(coeffs, domain, vf, xis_of)
        self.assertTrue(bool(jnp.all(got == want)), "wrapped-opening mismatch")


class OpenColumnsTest(absltest.TestCase):
    def test_recovers_direct_eval(self) -> None:
        # Barycentric round trip: Σ_i L_i(z)·p(domain[i]) == p(z), one base + one
        # cubic column, split as the composition consumes them.
        n = 1 << _N_BITS
        domain = eval_domain(KB, n)
        z = _rand_ext(1, 42)[0]
        weights = compute_lagrange_basis(z, domain.astype(KB4))[:, None]  # (N, 1)
        base_coeffs = _rand_base(n, 3)
        cubic_coeffs = _rand_ext(n, 4)
        base_col = _evals_on(base_coeffs, domain)[:, None]  # (N, 1) base
        cubic_col = _evals_on(cubic_coeffs, domain)[:, None]  # (N, 1) cubic
        got = open_columns(base_col, cubic_col, weights, [0, 0], stride=1)
        self.assertEqual(_limbs(got[0]), _limbs(_poly_eval(base_coeffs, z)))
        self.assertEqual(_limbs(got[1]), _limbs(_poly_eval(cubic_coeffs, z)))


if __name__ == "__main__":
    absltest.main()
