# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end self-verification of the dense LogUp-GKR prover.

zorch has no production GKR verifier yet, but the per-layer sumcheck reuses the
shared `verify` / `SumcheckVerifier`. This test drives each layer through them,
checks the LogUp oracle per layer via `LogupGkrRound._combine`, and confirms the
reduced claim lands on the input layer's leaf MLE (GKR completeness).
"""

from dataclasses import replace

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.circuit import _interleave
from zorch.logup_gkr.prove import _eq_weights, eval_mle, prove_logup_gkr
from zorch.logup_gkr.round import LogupGkrRound
from zorch.logup_gkr.testing import random_first_layer
from zorch.sumcheck import SumcheckVerifier
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript
from zorch.verify import verify

KB = zk_dtypes.koalabear
_DEGREE = 3


def _verify_gkr(output, proof, transcript):
    """Drive each layer through the shared verifier; reduce across the tree.

    Returns (num_eval, den_eval, point, ok). `ok` ANDs every layer's sumcheck
    check and LogUp oracle check.
    """
    initial_num_vars = output.numerator.shape[0].bit_length() - 1
    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(output.denominator)
    transcript, eval_point = transcript.sample(initial_num_vars)
    num_eval = eval_mle(output.numerator, eval_point)
    den_eval = eval_mle(output.denominator, eval_point)
    ok = True

    for lp in proof.layer_proofs:
        transcript, lam = transcript.sample(1)
        lam = lam[0]
        claim = lam * num_eval + den_eval

        point, final_claim, transcript, ok_sc = verify(
            SumcheckVerifier(_DEGREE), claim, lp.round_polys, transcript
        )
        ok = ok and bool(ok_sc)

        # LogUp oracle: the reduced claim equals the combine at the bound point.
        # eq is flipped because the rounds bind MSB-first (see prove.py).
        eq_eval = eval_mle(_eq_weights(eval_point), jnp.flip(point))
        combined = LogupGkrRound(lam)._combine(
            eq_eval, lp.numerator_0, lp.denominator_1, lp.numerator_1, lp.denominator_0
        )
        ok = ok and bool(combined == final_claim)

        transcript = transcript.observe(
            jnp.stack(
                [lp.numerator_0, lp.numerator_1, lp.denominator_0, lp.denominator_1]
            )
        )
        transcript, r = transcript.sample(1)
        r = r[0]
        num_eval = lp.numerator_0 + (lp.numerator_1 - lp.numerator_0) * r
        den_eval = lp.denominator_0 + (lp.denominator_1 - lp.denominator_0) * r
        eval_point = jnp.concatenate([jnp.atleast_1d(r), jnp.flip(point)])

    return num_eval, den_eval, eval_point, ok


class ProveTest(absltest.TestCase):
    def _check_roundtrip(self, first, num_layers, seed):
        ch = rand_field(seed, (128,), KB)
        _, proof = prove_logup_gkr(first, StubTranscript(ch))
        self.assertEqual(len(proof.layer_proofs), num_layers)

        num_eval, den_eval, point, ok = _verify_gkr(
            proof.output, proof, StubTranscript(ch)
        )
        self.assertTrue(ok)

        # GKR completeness: the reduction lands on the input leaf MLE.
        leaf_num = _interleave(first.numerator_0, first.numerator_1)
        leaf_den = _interleave(first.denominator_0, first.denominator_1)
        self.assertEqual(point.shape[0], leaf_num.shape[0].bit_length() - 1)
        self.assertTrue(bool(num_eval == eval_mle(leaf_num, point)))
        self.assertTrue(bool(den_eval == eval_mle(leaf_den, point)))

    def test_self_verifies_and_reduces_to_input_mle(self):
        self._check_roundtrip(random_first_layer(7, 1, 2), num_layers=2, seed=99)

    def test_self_verifies_wider_layer(self):
        self._check_roundtrip(random_first_layer(11, 2, 3), num_layers=3, seed=123)

    def test_verify_rejects_tampered_round_poly(self):
        first = random_first_layer(7, 1, 2)
        ch = rand_field(99, (64,), KB)
        _, proof = prove_logup_gkr(first, StubTranscript(ch))
        bad = proof.layer_proofs[0]
        proof.layer_proofs[0] = replace(
            bad, round_polys=bad.round_polys.at[0, 0].add(jnp.array(1, KB))
        )
        _, _, _, ok = _verify_gkr(proof.output, proof, StubTranscript(ch))
        self.assertFalse(ok)


if __name__ == "__main__":
    absltest.main()
