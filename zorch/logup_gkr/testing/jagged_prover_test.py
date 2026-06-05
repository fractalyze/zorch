# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for the jagged LogUp-GKR prover.

Every check is anchored against the brute-force virtual-dense expansion
(`testing.virtual_planes`): the prover's closed-form virtual-mass corrections
must reproduce what a full `2^(niv+nrv)` materialization would sum. The
prove->verify roundtrip lives in jagged_verifier_test; this pins the prover's
own invariants.
"""

from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _interleave,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import (
    JaggedGkrLayerRound,
    JaggedLayerProof,
    prove_jagged_layer,
)
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.poly.univariate import eval_coeffs
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _logup_combine(
    lam: Array, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array
) -> Array:
    """The LogUp summand, restated locally: the oracle must not import the
    production combine, or a drift there would leave these tests silently
    self-consistent."""
    return eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)


def _virtual_claim(layer: JaggedGkrLayer, nrv: int, lam: Array, z: Array) -> Array:
    """Brute-force `sum_x eq(z, x) * combine(x)` over the virtual hypercube."""
    n0, n1, d0, d1 = virtual_planes(layer, nrv)
    eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
    return jnp.sum(_logup_combine(lam, eq, n0, d1, n1, d0))


class ProveJaggedLayerTest(absltest.TestCase):
    # Jagged shape exercising all padding paths: odd segments (3, 5), an
    # already-saturated segment (1), and an even one (2); nrv = 3 embeds the
    # max count 5 into 8 virtual rows.
    ROW_COUNTS = (3, 1, 5, 2)
    NRV = 3

    def _prove(
        self, seed: int = 7
    ) -> tuple[JaggedGkrLayer, Array, Array, Array, Array, JaggedLayerProof]:
        layer = random_jagged_layer(seed, self.ROW_COUNTS)
        lam = rand_field(seed + 10, (), KB)
        z = rand_field(seed + 11, (self.NRV + 2,), KB)
        claim = _virtual_claim(layer, self.NRV, lam, z)
        point, _, proof = prove_jagged_layer(layer, lam, claim, z, cheap_transcript(KB))
        return layer, lam, z, claim, point, proof

    def test_first_round_matches_virtual_hypercube_claim(self) -> None:
        _, _, _, claim, _, proof = self._prove()
        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(bool(eval_coeffs(p0, zero) + eval_coeffs(p0, one) == claim))

    def test_round_polys_thread_claims(self) -> None:
        # Each round's s(0) + s(1) must equal the previous round's claim
        # reduction -- the per-round sumcheck identity over all five rounds.
        _, _, _, claim, point, proof = self._prove()
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            self.assertTrue(
                bool(eval_coeffs(poly, zero) + eval_coeffs(poly, one) == claim)
            )
            claim = eval_coeffs(poly, r)

    def test_final_openings_are_virtual_plane_evals(self) -> None:
        layer, _, _, _, point, proof = self._prove()
        n0, n1, d0, d1 = virtual_planes(layer, self.NRV)
        for got, plane in (
            (proof.numerator_0, n0),
            (proof.numerator_1, n1),
            (proof.denominator_0, d0),
            (proof.denominator_1, d1),
        ):
            self.assertTrue(bool(got == eval_mle(plane, point)))

    def test_final_claim_matches_oracle(self) -> None:
        # The fully-reduced claim equals the combine at the bound point with
        # eq evaluated in closed form -- the verifier's final check.
        _, lam, z, claim, point, proof = self._prove()
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            claim = eval_coeffs(poly, r)
        want = _logup_combine(
            lam,
            eval_eq(z, point),
            proof.numerator_0,
            proof.denominator_1,
            proof.numerator_1,
            proof.denominator_0,
        )
        self.assertTrue(bool(claim == want))

    def test_saturated_layer_folds_pure_virtual_rows(self) -> None:
        # One materialized row per segment under nrv=3: seven of every eight
        # virtual rows are non-materialized, so the closed-form corrections
        # carry almost the whole sum.
        layer = random_jagged_layer(21, (1, 1, 1, 1))
        lam = rand_field(31, (), KB)
        z = rand_field(32, (5,), KB)
        claim = _virtual_claim(layer, 3, lam, z)
        point, _, proof = prove_jagged_layer(layer, lam, claim, z, cheap_transcript(KB))
        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(bool(eval_coeffs(p0, zero) + eval_coeffs(p0, one) == claim))
        n0, _, _, d1 = virtual_planes(layer, 3)
        self.assertTrue(bool(proof.numerator_0 == eval_mle(n0, point)))
        self.assertTrue(bool(proof.denominator_1 == eval_mle(d1, point)))

    def test_multi_limb_challenges_extend_the_transcript_field(self) -> None:
        # A base-field transcript with extension-field claims: every
        # challenge takes four squeezes reinterpreted as the extension
        # element. The oracle identity must close in the extension field.
        layer = random_jagged_layer(41, self.ROW_COUNTS)
        lam = rand_ext_field(51, (), KB, EF)
        z = rand_ext_field(52, (self.NRV + 2,), KB, EF)
        claim = _virtual_claim(layer, self.NRV, lam, z)
        point, _, proof = prove_jagged_layer(
            layer, lam, claim, z, cheap_transcript(KB), challenge_limbs=4
        )
        self.assertEqual(point.dtype, EF)
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            claim = eval_coeffs(poly, r)
        want = _logup_combine(
            lam,
            eval_eq(z, point),
            proof.numerator_0,
            proof.denominator_1,
            proof.numerator_1,
            proof.denominator_0,
        )
        self.assertTrue(bool(claim == want))

    def test_rejects_missing_row_variable(self) -> None:
        layer = random_jagged_layer(61, (1, 1))
        with self.assertRaises(ValueError):
            prove_jagged_layer(
                layer,
                jnp.array(3, KB),
                jnp.array(5, KB),
                rand_field(62, (1,), KB),  # only the interaction variable
                cheap_transcript(KB),
            )

    def test_rejects_row_count_beyond_virtual_space(self) -> None:
        layer = random_jagged_layer(71, (5, 1))
        with self.assertRaises(ValueError):
            prove_jagged_layer(
                layer,
                jnp.array(3, KB),
                jnp.array(5, KB),
                rand_field(72, (3,), KB),  # nrv = 2 < log2(5)
                cheap_transcript(KB),
            )


class JaggedGkrLayerRoundTest(absltest.TestCase):
    def test_carry_reduction_closes_the_gkr_relation(self) -> None:
        # claim = lam * N(z) + D(z) with N/D read off the *folded* layer's
        # interleaved virtual planes must reduce, through the sumcheck, to
        # the input layer's interleaved planes at the new carry point -- the
        # cross-layer GKR identity on jagged layers.
        row_counts = (3, 1, 5, 2)
        nrv = 3
        layer = random_jagged_layer(81, row_counts)
        schedule = (2, 2, 4, 2)  # over-pads some segments past the fold
        folded = jagged_layer_transition(layer, schedule)

        z = rand_field(91, (nrv + 2,), KB)
        fn0, fn1, fd0, fd1 = virtual_planes(folded, nrv - 1)
        carry = (
            eval_mle(_interleave(fn0, fn1), z),
            eval_mle(_interleave(fd0, fd1), z),
            z,
        )
        transcript = cheap_transcript(KB)
        # Peek lam off the same transcript state (sample is non-destructive
        # on this one; the round re-derives it).
        _, lam = sample_challenge(transcript, KB, 1)

        (num_eval, den_eval, new_point), _, proof = JaggedGkrLayerRound(layer)(
            carry, transcript
        )

        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(
            bool(
                eval_coeffs(p0, zero) + eval_coeffs(p0, one)
                == lam * carry[0] + carry[1]
            )
        )
        self.assertEqual(new_point.shape, (nrv + 3,))
        n0, n1, d0, d1 = virtual_planes(layer, nrv)
        self.assertTrue(bool(num_eval == eval_mle(_interleave(n0, n1), new_point)))
        self.assertTrue(bool(den_eval == eval_mle(_interleave(d0, d1), new_point)))


if __name__ == "__main__":
    absltest.main()
