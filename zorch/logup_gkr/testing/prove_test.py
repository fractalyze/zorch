# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end self-verification of the dense LogUp-GKR prover.

zorch has no production verifier yet (the sumcheck round is prove-only and
checked by replay -- see sumcheck/testing/round_test.py). This test replays the
proof the same way: recompute the Fiat-Shamir challenges, check the per-round
sumcheck identity and the final combine, then confirm the reduced claim equals
the input layer's leaf MLE at the derived point (GKR completeness).
"""

from dataclasses import replace

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.circuit import GkrLayer, _interleave
from zorch.logup_gkr.prove import _eq_weights, prove_logup_gkr
from zorch.testkit.poly import eval_univariate
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


def _eval_mle(evals, point):
    """Multilinear extension of `evals` evaluated at `point` (LSB-first)."""
    return jnp.sum(evals * _eq_weights(point))


def _eval_eq(z, x):
    """eq(z, x) = prod_i (z_i*x_i + (1-z_i)(1-x_i)); z, x aligned coordinate-wise."""
    one = jnp.ones((), dtype=z.dtype)
    return jnp.prod(z * x + (one - z) * (one - x))


def _first_layer(seed, num_int_vars, num_row_vars):
    width = 1 << (num_int_vars + num_row_vars)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), KB),
        numerator_1=rand_field(seed + 1, (width,), KB),
        denominator_0=rand_field(seed + 2, (width,), KB),
        denominator_1=rand_field(seed + 3, (width,), KB),
        num_interaction_variables=num_int_vars,
    )


def _replay(output, proof, transcript):
    """Verify by replaying Fiat-Shamir; return (num_eval, den_eval, point, ok).

    Mirrors the prover's transcript discipline exactly, so the recomputed
    challenges match. `ok` is the conjunction of every per-round and final
    consistency check.
    """
    initial_num_vars = output.numerator.shape[0].bit_length() - 1
    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(output.denominator)
    transcript, z = transcript.sample(initial_num_vars)
    num_eval = _eval_mle(output.numerator, z)
    den_eval = _eval_mle(output.denominator, z)
    eval_point = z
    ok = True

    for lp in proof.layer_proofs:
        transcript, lam = transcript.sample(1)
        lam = lam[0]
        claim = lam * num_eval + den_eval

        chals = []
        for i in range(lp.round_polys.shape[0]):
            msg = lp.round_polys[i]
            ok = ok and bool(msg[0] + msg[1] == claim)  # s(0) + s(1) == claim
            transcript = transcript.observe(msg)
            transcript, c = transcript.sample(1)
            chals.append(c[0])
            claim = eval_univariate(msg, c[0])
        # MSB-first rounds: flip to index-order so coordinate j pairs with bit j.
        challenges = jnp.flip(jnp.stack(chals))

        eq_eval = _eval_eq(eval_point, challenges)
        n0, n1 = lp.numerator_0, lp.numerator_1
        d0, d1 = lp.denominator_0, lp.denominator_1
        combined = eq_eval * (lam * (n0 * d1 + n1 * d0) + d0 * d1)
        ok = ok and bool(combined == claim)

        transcript = transcript.observe(jnp.stack([n0, n1, d0, d1]))
        transcript, r = transcript.sample(1)
        r = r[0]
        num_eval = n0 + (n1 - n0) * r
        den_eval = d0 + (d1 - d0) * r
        eval_point = jnp.concatenate([jnp.atleast_1d(r), challenges])

    return num_eval, den_eval, eval_point, ok


class ProveTest(absltest.TestCase):
    def test_self_verifies_and_reduces_to_input_mle(self):
        first = _first_layer(7, num_int_vars=1, num_row_vars=2)
        challenges = rand_field(99, (64,), KB)

        _, proof = prove_logup_gkr(first, StubTranscript(challenges))
        # One sumcheck layer per row variable folded.
        self.assertEqual(len(proof.layer_proofs), 2)

        num_eval, den_eval, point, ok = _replay(
            proof.output, proof, StubTranscript(challenges)
        )
        self.assertTrue(ok)

        # GKR completeness: the reduced claim is the input leaf MLE at the
        # derived point.
        leaf_num = _interleave(first.numerator_0, first.numerator_1)
        leaf_den = _interleave(first.denominator_0, first.denominator_1)
        self.assertEqual(point.shape[0], leaf_num.shape[0].bit_length() - 1)
        self.assertTrue(bool(num_eval == _eval_mle(leaf_num, point)))
        self.assertTrue(bool(den_eval == _eval_mle(leaf_den, point)))

    def test_self_verifies_wider_layer(self):
        first = _first_layer(11, num_int_vars=2, num_row_vars=3)
        challenges = rand_field(123, (128,), KB)
        _, proof = prove_logup_gkr(first, StubTranscript(challenges))
        self.assertEqual(len(proof.layer_proofs), 3)
        num_eval, den_eval, point, ok = _replay(
            proof.output, proof, StubTranscript(challenges)
        )
        self.assertTrue(ok)
        leaf_num = _interleave(first.numerator_0, first.numerator_1)
        leaf_den = _interleave(first.denominator_0, first.denominator_1)
        self.assertTrue(bool(num_eval == _eval_mle(leaf_num, point)))
        self.assertTrue(bool(den_eval == _eval_mle(leaf_den, point)))

    def test_replay_rejects_tampered_round_poly(self):
        first = _first_layer(7, num_int_vars=1, num_row_vars=2)
        challenges = rand_field(99, (64,), KB)
        _, proof = prove_logup_gkr(first, StubTranscript(challenges))

        # Corrupt one round-polynomial coefficient; the sumcheck identity
        # (and thus replay) must reject.
        bad = proof.layer_proofs[0]
        polys = bad.round_polys.at[0, 0].add(jnp.array(1, KB))
        proof.layer_proofs[0] = replace(bad, round_polys=polys)
        _, _, _, ok = _replay(proof.output, proof, StubTranscript(challenges))
        self.assertFalse(ok)


if __name__ == "__main__":
    absltest.main()
