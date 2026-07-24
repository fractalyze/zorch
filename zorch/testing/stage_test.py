# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Claim-reduction Stage contract and explicit composite dataflow."""

from __future__ import annotations

from dataclasses import dataclass, replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


class _ScaleStage(Stage[Array, Array, Array, Array]):
    """Prove ``claim == witness`` conditional on a scaled claim."""

    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def prove(
        self, claim: Array, witness: Array, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        transcript, sampled = transcript.sample(1)
        reduced_claim = witness * self.factor + sampled[0]
        return ProveResult(reduced_claim, witness, transcript)

    def verify(
        self, claim: Array, reduction_proof: Array, transcript: Transcript
    ) -> VerifyResult[Array]:
        transcript, sampled = transcript.sample(1)
        reduced_claim = reduction_proof * self.factor + sampled[0]
        return VerifyResult(reduced_claim, transcript, reduction_proof == claim)


@dataclass(frozen=True)
class _MixClaim:
    original: Array
    scaled: Array


class _MixStage(Stage[_MixClaim, None, Array, Array]):
    """A reduction whose claim contains a skip-level public dependency."""

    def prove(
        self, claim: _MixClaim, witness: None, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        del witness
        return ProveResult(claim.original + claim.scaled, claim.scaled, transcript)

    def verify(
        self,
        claim: _MixClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        reduced_claim = claim.original + reduction_proof
        return VerifyResult(reduced_claim, transcript, reduction_proof == claim.scaled)


@dataclass(frozen=True)
class _CompositeProof:
    scale: Array
    mix: Array


class _CompositeStage(Stage[Array, Array, Array, _CompositeProof]):
    """Keep the root claim live while composing two child reductions."""

    def __init__(self) -> None:
        self.scale = _ScaleStage(3)
        self.mix = _MixStage()

    def prove(
        self, claim: Array, witness: Array, transcript: Transcript
    ) -> ProveResult[Array, _CompositeProof]:
        scaled = self.scale.prove(claim, witness, transcript)
        mixed = self.mix.prove(
            _MixClaim(original=claim, scaled=scaled.reduced_claim),
            None,
            scaled.transcript,
        )
        return ProveResult(
            mixed.reduced_claim,
            _CompositeProof(
                scale=scaled.reduction_proof,
                mix=mixed.reduction_proof,
            ),
            mixed.transcript,
        )

    def verify(
        self,
        claim: Array,
        reduction_proof: _CompositeProof,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        scaled = self.scale.verify(claim, reduction_proof.scale, transcript)
        mixed = self.mix.verify(
            _MixClaim(original=claim, scaled=scaled.reduced_claim),
            reduction_proof.mix,
            scaled.transcript,
        )
        return VerifyResult(mixed.reduced_claim, mixed.transcript, scaled.ok & mixed.ok)


@dataclass(frozen=True)
class _FoldClaim:
    accumulator: Array
    instances: Array


class _FoldStage(Stage[_FoldClaim, None, Array, Array]):
    """One k-ary accumulator claim reduction."""

    def prove(
        self, claim: _FoldClaim, witness: None, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        del witness
        delta = fnp.sum(claim.instances)
        return ProveResult(claim.accumulator + delta, delta, transcript)

    def verify(
        self,
        claim: _FoldClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        delta = fnp.sum(claim.instances)
        return VerifyResult(
            claim.accumulator + reduction_proof,
            transcript,
            reduction_proof == delta,
        )


class StageTest(absltest.TestCase):
    def test_composite_threads_named_claim_dependencies(self) -> None:
        protocol = _CompositeStage()
        initial = fnp.array(11, KB)

        proved = protocol.prove(initial, initial, cheap_transcript(KB))
        verified = protocol.verify(
            initial, proved.reduction_proof, cheap_transcript(KB)
        )

        self.assertTrue(bool(verified.ok))
        self.assertTrue(bool(verified.reduced_claim == proved.reduced_claim))

    def test_composite_rejects_tampered_child_proof(self) -> None:
        protocol = _CompositeStage()
        initial = fnp.array(9, KB)
        proved = protocol.prove(initial, initial, cheap_transcript(KB))
        bad = replace(
            proved.reduction_proof,
            mix=proved.reduction_proof.mix + fnp.array(1, KB),
        )

        verified = protocol.verify(initial, bad, cheap_transcript(KB))
        self.assertFalse(bool(verified.ok))

    def test_stage_can_thread_a_k_ary_folding_claim(self) -> None:
        stage = _FoldStage()
        prover_acc = fnp.array(0, KB)
        verifier_acc = fnp.array(0, KB)
        transcript_p: Transcript = cheap_transcript(KB)
        transcript_v: Transcript = cheap_transcript(KB)
        batches = (fnp.array([1, 2], KB), fnp.array([3, 4, 5], KB))

        for instances in batches:
            prover_claim = _FoldClaim(prover_acc, instances)
            verifier_claim = _FoldClaim(verifier_acc, instances)
            proved = stage.prove(prover_claim, None, transcript_p)
            verified = stage.verify(
                verifier_claim, proved.reduction_proof, transcript_v
            )
            prover_acc, transcript_p = proved.reduced_claim, proved.transcript
            verifier_acc, transcript_v = (
                verified.reduced_claim,
                verified.transcript,
            )
            self.assertTrue(bool(verified.ok))

        self.assertTrue(bool(prover_acc == verifier_acc))


if __name__ == "__main__":
    absltest.main()
