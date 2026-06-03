# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end self-verification of dense LogUp-GKR via the chained Rounds.

Prove with a ProveChain of layer rounds, verify with a VerifyChain of layer
rounds over the same transcript, and confirm: the proof self-verifies, the prover
and verifier thread the same reduction, the claim reduces onto the input leaf MLE
(GKR completeness), and a tampered round polynomial is rejected.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.circuit import GkrLayer, _interleave
from zorch.logup_gkr.testing import prove_gkr, random_first_layer, verify_gkr
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear


class GkrRoundtripTest(absltest.TestCase):
    def _roundtrip(self, first: GkrLayer, expected_layers: int, seed: int) -> None:
        ch = rand_field(seed, (128,), KB)
        _, output, proofs, prover_final = prove_gkr(first, ch)
        self.assertEqual(len(proofs), expected_layers)

        verifier_final, ok = verify_gkr(output, proofs, ch)
        self.assertTrue(bool(ok))

        # Prover and verifier thread identical reductions.
        for pe, ve in zip(prover_final, verifier_final, strict=True):
            self.assertTrue(bool(jnp.all(pe == ve)))

        # GKR completeness: the reduction lands on the input leaf MLE.
        num_eval, den_eval, point = verifier_final
        leaf_num = _interleave(first.numerator_0, first.numerator_1)
        leaf_den = _interleave(first.denominator_0, first.denominator_1)
        self.assertEqual(point.shape[0], first.num_variables + 1)
        self.assertTrue(bool(num_eval == eval_mle(leaf_num, point)))
        self.assertTrue(bool(den_eval == eval_mle(leaf_den, point)))

    def test_self_verifies_two_layers(self) -> None:
        self._roundtrip(random_first_layer(7, 1, 2), expected_layers=2, seed=99)

    def test_self_verifies_wider(self) -> None:
        self._roundtrip(random_first_layer(11, 2, 3), expected_layers=3, seed=123)

    def test_rejects_tampered_round_poly(self) -> None:
        first = random_first_layer(7, 1, 2)
        ch = rand_field(99, (128,), KB)
        _, output, proofs, _ = prove_gkr(first, ch)
        bad = proofs[0]
        proofs[0] = replace(
            bad, round_polys=bad.round_polys.at[0, 0].add(jnp.array(1, KB))
        )
        _, ok = verify_gkr(output, proofs, ch)
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
