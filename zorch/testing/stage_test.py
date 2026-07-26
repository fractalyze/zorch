# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable Stage roles and explicit composite dataflow."""

from __future__ import annotations

from dataclasses import dataclass, replace

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.stage import (
    ProveResult,
    ProverStage,
    VerifierStage,
    VerifyResult,
)
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


class _ScaleProver(ProverStage[Array, Array, Array, Array]):
    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def prove(
        self, claim: Array, witness: Array, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        del claim
        transcript, sampled = transcript.sample(1)
        return ProveResult(witness * self.factor + sampled[0], witness, transcript)


class _ScaleVerifier(VerifierStage[Array, Array, Array]):
    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def verify(
        self, claim: Array, reduction_proof: Array, transcript: Transcript
    ) -> VerifyResult[Array]:
        transcript, sampled = transcript.sample(1)
        reduced = reduction_proof * self.factor + sampled[0]
        return VerifyResult(reduced, transcript, reduction_proof == claim)


@dataclass(frozen=True)
class _MixClaim:
    original: Array
    scaled: Array


class _MixProver(ProverStage[_MixClaim, None, Array, Array]):
    def prove(
        self, claim: _MixClaim, witness: None, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        del witness
        return ProveResult(claim.original + claim.scaled, claim.scaled, transcript)


class _MixVerifier(VerifierStage[_MixClaim, Array, Array]):
    def verify(
        self,
        claim: _MixClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        return VerifyResult(
            claim.original + reduction_proof,
            transcript,
            reduction_proof == claim.scaled,
        )


@dataclass(frozen=True)
class _CompositeProof:
    scale: Array
    mix: Array


class _CompositeProver(ProverStage[Array, Array, Array, _CompositeProof]):
    def __init__(self) -> None:
        self.scale = _ScaleProver(3)
        self.mix = _MixProver()

    def prove(
        self, claim: Array, witness: Array, transcript: Transcript
    ) -> ProveResult[Array, _CompositeProof]:
        scaled = self.scale.prove(claim, witness, transcript)
        mixed = self.mix.prove(
            _MixClaim(claim, scaled.reduced_claim), None, scaled.transcript
        )
        return ProveResult(
            mixed.reduced_claim,
            _CompositeProof(scaled.reduction_proof, mixed.reduction_proof),
            mixed.transcript,
        )


class _CompositeVerifier(VerifierStage[Array, Array, _CompositeProof]):
    def __init__(self) -> None:
        self.scale = _ScaleVerifier(3)
        self.mix = _MixVerifier()

    def verify(
        self,
        claim: Array,
        reduction_proof: _CompositeProof,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        scaled = self.scale.verify(claim, reduction_proof.scale, transcript)
        mixed = self.mix.verify(
            _MixClaim(claim, scaled.reduced_claim),
            reduction_proof.mix,
            scaled.transcript,
        )
        return VerifyResult(mixed.reduced_claim, mixed.transcript, scaled.ok & mixed.ok)


@dataclass(frozen=True)
class _FoldClaim:
    accumulator: Array
    instances: Array


class _FoldProver(ProverStage[_FoldClaim, None, Array, Array]):
    def prove(
        self, claim: _FoldClaim, witness: None, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        del witness
        delta = fnp.sum(claim.instances)
        return ProveResult(claim.accumulator + delta, delta, transcript)


class _FoldVerifier(VerifierStage[_FoldClaim, Array, Array]):
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
    def test_composite_rejects_tampered_child_proof(self) -> None:
        prover = _CompositeProver()
        verifier = _CompositeVerifier()
        initial = fnp.array(9, KB)
        proved = prover.prove(initial, initial, cheap_transcript(KB))
        bad = replace(
            proved.reduction_proof,
            mix=proved.reduction_proof.mix + fnp.array(1, KB),
        )
        self.assertFalse(bool(verifier.verify(initial, bad, cheap_transcript(KB)).ok))

    def test_roles_thread_a_k_ary_folding_claim(self) -> None:
        prover = _FoldProver()
        verifier = _FoldVerifier()
        prover_acc = fnp.array(0, KB)
        verifier_acc = fnp.array(0, KB)
        transcript_p: Transcript = cheap_transcript(KB)
        transcript_v: Transcript = cheap_transcript(KB)
        batches = (fnp.array([1, 2], KB), fnp.array([3, 4, 5], KB))

        for instances in batches:
            proved = prover.prove(_FoldClaim(prover_acc, instances), None, transcript_p)
            verified = verifier.verify(
                _FoldClaim(verifier_acc, instances),
                proved.reduction_proof,
                transcript_v,
            )
            prover_acc, transcript_p = proved.reduced_claim, proved.transcript
            verifier_acc, transcript_v = verified.reduced_claim, verified.transcript
            self.assertTrue(bool(verified.ok))

        self.assertTrue(bool(prover_acc == verifier_acc))


if __name__ == "__main__":
    absltest.main()
