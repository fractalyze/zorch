# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Prove -> verify roundtrip over a jagged GKR pyramid.

The chain runs a mixed schedule -- over-padded segments next to saturated
ones -- so every verifier layer replays round polynomials whose virtual-mass
corrections actually carried weight. The roundtrip closes against the input
layer's virtual dense leaf MLE, the check a PCS opening performs in
production.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    LogUpGkrOutput,
    _interleave,
    extract_jagged_outputs,
)
from zorch.logup_gkr.jagged_prover import JaggedGkrLayerRound, JaggedLayerProof
from zorch.logup_gkr.jagged_stage import (
    JaggedGkrWitness,
    JaggedLogUpGkrProver,
    JaggedLogUpGkrVerifier,
)
from zorch.logup_gkr.jagged_verifier import (
    JaggedGkrLayerRound as JaggedVerifierLayerRound,
)
from zorch.logup_gkr.prover import LayerClaim, bind_output
from zorch.logup_gkr.stage import LogUpOutputClaim
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    caps_for,
    host_counts,
    jagged_fold_schedules,
    random_jagged_layer,
    virtual_planes,
)
from zorch.poly.multilinear import eval_mle
from zorch.round import prove_rounds, verify_rounds
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont

# The transcript's own field: the schedule these tests pinned before the
# policy required an explicit field.
_CH = ChallengePolicy(KB)
EF = zk_dtypes.koalabearx4_mont

ROW_COUNTS = (3, 1, 5, 2)


def _prove(
    layers: list[JaggedGkrLayer],
) -> tuple[LayerClaim, list[JaggedLayerProof], LogUpGkrOutput]:
    output = extract_jagged_outputs(layers[-1])
    carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)
    caps = caps_for(host_counts(layers[0]), len(layers) - 1)
    final, _, proofs = prove_rounds(
        [
            JaggedGkrLayerRound(layer, caps=caps, challenges=_CH)
            for layer in reversed(layers[:-1])
        ],
        carry,
        transcript,
    )
    return final, proofs, output


def _verify(
    output: LogUpGkrOutput, proofs: list[JaggedLayerProof]
) -> tuple[LayerClaim, Array]:
    carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)
    final, _, ok = verify_rounds(
        [JaggedVerifierLayerRound(_CH) for _ in proofs], carry, proofs, transcript
    )
    return final, ok


class JaggedRoundtripTest(absltest.TestCase):
    def test_roundtrip_accepts_and_closes_on_leaf_mle(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, ROW_COUNTS))
        prover_final, proofs, output = _prove(layers)
        self.assertEqual(len(proofs), len(layers) - 1)

        verifier_final, ok = _verify(output, proofs)
        self.assertTrue(bool(ok))
        # Fiat-Shamir lockstep: both sides reduce to the same point-claim.
        for got, want in zip(verifier_final, prover_final, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))

        # The PCS-consumer closing check, done directly: the reduced claims
        # are the input layer's interleaved virtual planes at the final point.
        num_eval, den_eval, point = verifier_final
        nrv = point.shape[0] - layers[0].num_batch_variables - 1
        n0, n1, d0, d1 = virtual_planes(layers[0], nrv)
        self.assertTrue(bool(num_eval == eval_mle(_interleave(n0, n1), point)))
        self.assertTrue(bool(den_eval == eval_mle(_interleave(d0, d1), point)))

    def test_tampered_round_poly_rejected(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(17, ROW_COUNTS))
        _, proofs, output = _prove(layers)
        bumped = proofs[1].round_polys.at[0, 0].add(fnp.array(1, KB))
        proofs[1] = replace(proofs[1], round_polys=bumped)
        _, ok = _verify(output, proofs)
        self.assertFalse(bool(ok))

    def test_tampered_opening_rejected(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(27, ROW_COUNTS))
        _, proofs, output = _prove(layers)
        bad = proofs[0].numerator_0 + fnp.array(1, KB)
        proofs[0] = replace(proofs[0], numerator_0=bad)
        _, ok = _verify(output, proofs)
        self.assertFalse(bool(ok))

    def test_multi_limb_roundtrip(self) -> None:
        # Extension-field claims over a base-field transcript: both sides
        # bind and squeeze through the same multi-limb rule.
        limbs = 4
        challenges = ChallengePolicy(EF)
        layers = build_jagged_pyramid(random_jagged_layer(37, ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])

        carry, transcript = bind_output(output, cheap_transcript(KB), challenges)
        prover_final, _, proofs = prove_rounds(
            [
                JaggedGkrLayerRound(
                    layer, challenges, caps=caps_for(ROW_COUNTS, len(layers) - 1)
                )
                for layer in reversed(layers[:-1])
            ],
            carry,
            transcript,
        )

        carry, transcript = bind_output(output, cheap_transcript(KB), challenges)
        verifier_final, _, ok = verify_rounds(
            [JaggedVerifierLayerRound(challenges) for _ in proofs],
            carry,
            proofs,
            transcript,
        )
        self.assertTrue(bool(ok))
        self.assertEqual(verifier_final[0].dtype, EF)
        for got, want in zip(verifier_final, prover_final, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))


class JaggedStageTest(absltest.TestCase):
    """The jagged Stage pair over the same pyramid the round chain runs."""

    def _fixture(self, seed: int) -> tuple[
        JaggedLogUpGkrProver,
        JaggedLogUpGkrVerifier,
        LogUpOutputClaim,
        JaggedGkrWitness,
        list[JaggedGkrLayer],
    ]:
        first = random_jagged_layer(seed, ROW_COUNTS)
        schedules = jagged_fold_schedules(first)
        layers = build_jagged_pyramid(first)
        claim = LogUpOutputClaim(extract_jagged_outputs(layers[-1]), len(schedules))
        return (
            JaggedLogUpGkrProver(
                caps_for(host_counts(first), len(schedules)), challenges=_CH
            ),
            JaggedLogUpGkrVerifier(challenges=_CH),
            claim,
            JaggedGkrWitness(first, schedules),
            layers,
        )

    def test_stage_roundtrips_and_transcripts_agree(self) -> None:
        prover, verifier, claim, witness, _ = self._fixture(7)
        proved = prover.prove(claim, witness, cheap_transcript(KB))
        verified = verifier.verify(claim, proved.reduction_proof, cheap_transcript(KB))
        self.assertTrue(bool(verified.ok))
        self.assertEqual(len(proved.reduction_proof.layers), claim.layers)
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

    def test_stage_matches_the_hand_run_chain(self) -> None:
        # The stage is a wrapper, so its stream must be the round chain's:
        # same layer proofs, same reduced claim, chain-owned buffers and all.
        prover, _, claim, witness, layers = self._fixture(17)
        hand_final, hand_proofs, _ = _prove(layers)
        proved = prover.prove(claim, witness, cheap_transcript(KB))
        for got, want in zip(proved.reduction_proof.layers, hand_proofs, strict=True):
            for field in ("round_polys", "point", "numerator_0", "denominator_1"):
                self.assertTrue(
                    bool(fnp.all(getattr(got, field) == getattr(want, field)))
                )
        num_eval, den_eval, point = hand_final
        self.assertTrue(bool(proved.reduced_claim.numerator == num_eval))
        self.assertTrue(bool(proved.reduced_claim.denominator == den_eval))
        self.assertTrue(bool(fnp.all(proved.reduced_claim.point == point)))

    def test_stage_statement_owns_layer_count(self) -> None:
        prover, verifier, claim, witness, _ = self._fixture(27)
        proved = prover.prove(claim, witness, cheap_transcript(KB))
        bad = replace(claim, layers=claim.layers + 1)
        with self.assertRaises(ValueError):
            verifier.verify(bad, proved.reduction_proof, cheap_transcript(KB))
        with self.assertRaises(ValueError):
            prover.prove(bad, witness, cheap_transcript(KB))

    def test_stage_rejects_tampered_round_poly(self) -> None:
        prover, verifier, claim, witness, _ = self._fixture(37)
        proved = prover.prove(claim, witness, cheap_transcript(KB))
        layer_proofs = list(proved.reduction_proof.layers)
        bad = layer_proofs[1]
        layer_proofs[1] = replace(
            bad, round_polys=bad.round_polys.at[0, 0].add(fnp.array(1, KB))
        )
        verified = verifier.verify(
            claim,
            replace(proved.reduction_proof, layers=tuple(layer_proofs)),
            cheap_transcript(KB),
        )
        self.assertFalse(bool(verified.ok))


if __name__ == "__main__":
    absltest.main()
