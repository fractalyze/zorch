# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end self-verification of dense LogUp-GKR via the chained Rounds.

Prove with a prove_rounds of layer rounds, verify with a verify_rounds of layer
rounds over the same transcript, and confirm: the proof self-verifies, the prover
and verifier thread the same reduction, the claim reduces onto the input leaf MLE
(GKR completeness), and a tampered round polynomial is rejected.
"""

from __future__ import annotations

from dataclasses import replace

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.challenge import ChallengePolicy
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.logup_gkr.circuit import (
    GkrLayer,
    _interleave,
    build_pyramid,
    extract_outputs,
)
from zorch.logup_gkr.stage import (
    LogUpGkrProver,
    LogUpGkrVerifier,
    LogUpOutputClaim,
)
from zorch.logup_gkr.testing import (
    prove_gkr,
    prove_gkr_with_transcript,
    random_first_layer,
    verify_gkr,
    verify_gkr_with_transcript,
)
from zorch.poly.multilinear import eval_mle
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)

# A poseidon2 permute is impractically slow to compile on the XLA CPU
# backend, so
# the real-transcript e2e is GPU-only. transcript_test gates its DuplexTranscript
# cases on the same CPU-backend check (for its own reason), so neither runs in the
# CPU CI lane.
_CPU_BACKEND = frx.default_backend() == "cpu"


class GkrRoundtripTest(absltest.TestCase):
    def _roundtrip(self, first: GkrLayer, expected_layers: int) -> None:
        _, output, proofs, prover_final = prove_gkr(first)
        self.assertEqual(len(proofs), expected_layers)

        verifier_final, ok = verify_gkr(output, proofs)
        self.assertTrue(bool(ok))

        # Prover and verifier thread identical reductions.
        for pe, ve in zip(prover_final, verifier_final, strict=True):
            self.assertTrue(bool(fnp.all(pe == ve)))

        # GKR completeness: the reduction lands on the input leaf MLE.
        num_eval, den_eval, point = verifier_final
        leaf_num = _interleave(first.numerator_0, first.numerator_1)
        leaf_den = _interleave(first.denominator_0, first.denominator_1)
        self.assertEqual(point.shape[0], first.num_variables + 1)
        self.assertTrue(bool(num_eval == eval_mle(leaf_num, point)))
        self.assertTrue(bool(den_eval == eval_mle(leaf_den, point)))

    def test_self_verifies_two_layers(self) -> None:
        self._roundtrip(random_first_layer(7, 1, 2), expected_layers=2)

    def test_self_verifies_wider(self) -> None:
        self._roundtrip(random_first_layer(11, 2, 3), expected_layers=3)

    def test_stage_roundtrips_and_transcripts_agree(self) -> None:
        first = random_first_layer(17, 1, 2)
        prover = LogUpGkrProver(challenges=_CH)
        verifier = LogUpGkrVerifier(challenges=_CH)
        claim = LogUpOutputClaim(
            extract_outputs(build_pyramid(first)[-1]), first.num_row_variables
        )
        proved = prover.prove(claim, first, cheap_transcript(KB))
        verified = verifier.verify(claim, proved.reduction_proof, cheap_transcript(KB))
        self.assertTrue(bool(verified.ok))
        self.assertTrue(
            bool(fnp.all(proved.reduced_claim.point == verified.reduced_claim.point))
        )
        self.assertTrue(
            bool(proved.reduced_claim.numerator == verified.reduced_claim.numerator)
        )
        self.assertTrue(
            bool(proved.reduced_claim.denominator == verified.reduced_claim.denominator)
        )
        _, prover_next = proved.transcript.sample(1)
        _, verifier_next = verified.transcript.sample(1)
        self.assertTrue(bool(fnp.all(prover_next == verifier_next)))

    def test_stage_statement_owns_layer_count(self) -> None:
        first = random_first_layer(19, 1, 2)
        prover = LogUpGkrProver(challenges=_CH)
        verifier = LogUpGkrVerifier(challenges=_CH)
        claim = LogUpOutputClaim(
            extract_outputs(build_pyramid(first)[-1]), first.num_row_variables
        )
        proved = prover.prove(claim, first, cheap_transcript(KB))
        bad = replace(claim, layers=claim.layers + 1)
        with self.assertRaises(ValueError):
            verifier.verify(bad, proved.reduction_proof, cheap_transcript(KB))

    def test_rejects_tampered_round_poly(self) -> None:
        first = random_first_layer(7, 1, 2)
        _, output, proofs, _ = prove_gkr(first)
        bad = proofs[0]
        proofs[0] = replace(
            bad, round_polys=bad.round_polys.at[0, 0].add(fnp.array(1, KB))
        )
        _, ok = verify_gkr(output, proofs)
        self.assertFalse(bool(ok))


@absltest.skipIf(_CPU_BACKEND, "poseidon2 compile is impractically slow on XLA CPU")
class GkrDuplexRoundtripTest(absltest.TestCase):
    """Self-verification through the real on-device poseidon2 DuplexTranscript:
    challenges are squeezed from the sponge (no preset stream), and the verifier
    re-derives the same stream from a fresh transcript with the same permutation.
    """

    @staticmethod
    def _transcript() -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def test_self_verifies_with_duplex_transcript(self) -> None:
        first = random_first_layer(7, 1, 2)
        _, output, proofs, prover_final = prove_gkr_with_transcript(
            first, self._transcript()
        )
        verifier_final, ok = verify_gkr_with_transcript(
            output, proofs, self._transcript()
        )
        self.assertTrue(bool(ok))
        for pe, ve in zip(prover_final, verifier_final, strict=True):
            self.assertTrue(bool(fnp.all(pe == ve)))


if __name__ == "__main__":
    absltest.main()
