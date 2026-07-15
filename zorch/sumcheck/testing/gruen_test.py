# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx
import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.poly.eq import eq_root
from zorch.poly.univariate import eval_coeffs
from zorch.sumcheck import gruen

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _poly_with_root_at_b(q_coeffs: Array, b: Array) -> Array:
    """Ascending coefficients of p(x) = (x - b)·q(x): a polynomial with the
    known zero at ``b`` every Gruen round poly carries."""
    dtype = q_coeffs.dtype
    zero = jnp.zeros((), dtype)
    shifted = jnp.concatenate([zero[None], q_coeffs])  # x·q(x)
    scaled = jnp.concatenate([q_coeffs * b, zero[None]])  # b·q(x)
    return shifted - scaled


class RoundCoeffsTest(absltest.TestCase):
    def _assert_recovers(self, extra_ts: list[Array], q_coeffs: Array) -> None:
        """round_coeffs on p's evaluations must recover p's coefficients."""
        dtype = q_coeffs.dtype
        z = jnp.array(9, dtype)
        p_coeffs = _poly_with_root_at_b(q_coeffs, eq_root(z))
        s_zero = eval_coeffs(p_coeffs, jnp.zeros((), dtype))
        claim = s_zero + eval_coeffs(p_coeffs, jnp.ones((), dtype))
        extra_ys = [eval_coeffs(p_coeffs, t) for t in extra_ts]
        got = gruen.round_coeffs(s_zero, claim, extra_ts, extra_ys, z)
        self.assertEqual(got.shape, p_coeffs.shape)
        self.assertTrue(bool(jnp.all(got == p_coeffs)))

    def test_logup_domain_degree3(self) -> None:
        # The LogUp jagged instance: extra point 1/2, degree 3.
        half = jnp.ones((), KB) / jnp.array(2, KB)
        self._assert_recovers([half], jnp.array([3, 1, 4], KB))

    def test_zerocheck_domain_degree4(self) -> None:
        # The zerocheck jagged instance: extra points {2, 4}, degree 4.
        two = jnp.array(2, KB)
        four = jnp.array(4, KB)
        self._assert_recovers([two, four], jnp.array([3, 1, 4, 1], KB))

    def test_zerocheck_domain_extension_field(self) -> None:
        # The zerocheck instance runs in an extension field.
        two = jnp.array(2, EF)
        four = jnp.array(4, EF)
        self._assert_recovers([two, four], jnp.array([7, 0, 2, 5], EF))

    def test_no_extra_points_degree2(self) -> None:
        # Degenerate instance: {0, 1, b} alone pins a degree-2 round poly.
        self._assert_recovers([], jnp.array([5, 2], KB))

    def test_matches_eager_under_jit(self) -> None:
        # The interpolation constants are built inside the call; under jit they
        # must resolve concretely (compile-time eval) and match eager bytes.
        half = jnp.ones((), KB) / jnp.array(2, KB)
        z = jnp.array(9, KB)
        p_coeffs = _poly_with_root_at_b(jnp.array([3, 1, 4], KB), eq_root(z))
        s_zero = eval_coeffs(p_coeffs, jnp.zeros((), KB))
        claim = s_zero + eval_coeffs(p_coeffs, jnp.ones((), KB))
        s_half = eval_coeffs(p_coeffs, half)

        def assemble(s0: Array, c: Array, y: Array, zz: Array) -> Array:
            return gruen.round_coeffs(s0, c, [half], [y], zz)

        eager = assemble(s_zero, claim, s_half, z)
        jitted = frx.jit(assemble)(s_zero, claim, s_half, z)
        self.assertTrue(bool(jnp.all(eager == jitted)))

    def test_mismatched_extra_lengths_raise(self) -> None:
        one = jnp.ones((), KB)
        with self.assertRaises(ValueError):
            gruen.round_coeffs(one, one, [one], [], jnp.array(9, KB))
        with self.assertRaises(ValueError):
            gruen.round_coeffs_from_matrix(
                gruen.interp_matrix([one], jnp.array(9, KB)), one, one, []
            )

    def test_precomputed_matrix_matches_composition(self) -> None:
        # A scan driver builds the per-round matrices outside the loop (vmap
        # over the stacked round coordinates); the split seam must agree with
        # the one-call composition exactly.
        two = jnp.array(2, KB)
        four = jnp.array(4, KB)
        zs = jnp.array([9, 13, 21], KB)
        matrices = frx.vmap(lambda z: gruen.interp_matrix([two, four], z))(zs)
        p = _poly_with_root_at_b(jnp.array([3, 1, 4, 1], KB), eq_root(zs[1]))
        s_zero = eval_coeffs(p, jnp.zeros((), KB))
        claim = s_zero + eval_coeffs(p, jnp.ones((), KB))
        ys = [eval_coeffs(p, two), eval_coeffs(p, four)]
        via_matrix = gruen.round_coeffs_from_matrix(matrices[1], s_zero, claim, ys)
        composed = gruen.round_coeffs(s_zero, claim, [two, four], ys, zs[1])
        self.assertTrue(bool(jnp.all(via_matrix == composed)))

    def test_batched_evaluations_broadcast(self) -> None:
        # Per-chip value sets ride a leading batch axis through the assembly:
        # column c of the batched result equals the scalar call for chip c.
        two = jnp.array(2, KB)
        four = jnp.array(4, KB)
        z = jnp.array(9, KB)
        matrix = gruen.interp_matrix([two, four], z)
        chips = []
        for seed in (jnp.array([3, 1, 4, 1], KB), jnp.array([7, 0, 2, 5], KB)):
            p = _poly_with_root_at_b(seed, eq_root(z))
            s0 = eval_coeffs(p, jnp.zeros((), KB))
            chips.append(
                (
                    s0,
                    s0 + eval_coeffs(p, jnp.ones((), KB)),
                    eval_coeffs(p, two),
                    eval_coeffs(p, four),
                )
            )
        s0s, claims, y2s, y4s = (jnp.stack(col) for col in zip(*chips))
        batched = gruen.round_coeffs_from_matrix(matrix, s0s, claims, [y2s, y4s])
        self.assertEqual(batched.shape, (5, 2))
        for c, (s0, claim, y2, y4) in enumerate(chips):
            single = gruen.round_coeffs_from_matrix(matrix, s0, claim, [y2, y4])
            self.assertTrue(bool(jnp.all(batched[:, c] == single)))


if __name__ == "__main__":
    absltest.main()
