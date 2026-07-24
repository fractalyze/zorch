# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Paired Stage contract and explicit composite-stage dataflow."""

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


class _ScaleStage(Stage[Array, Array, Array, Array, Array]):
    name = "scale"

    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def prove(self, inputs: Array, transcript: Transcript) -> ProveResult[Array, Array]:
        transcript, sampled = transcript.sample(1)
        return ProveResult(inputs * self.factor + sampled[0], inputs, transcript)

    def verify(
        self, inputs: Array, proof: Array, transcript: Transcript
    ) -> VerifyResult[Array]:
        transcript, sampled = transcript.sample(1)
        output = inputs * self.factor + sampled[0]
        return VerifyResult(output, transcript, proof == inputs)


@dataclass(frozen=True)
class _MixInput:
    original: Array
    scaled: Array


class _MixStage(Stage[_MixInput, Array, _MixInput, Array, Array]):
    """A child that needs a skip-level value and its predecessor's output."""

    name = "mix"

    def prove(
        self, inputs: _MixInput, transcript: Transcript
    ) -> ProveResult[Array, Array]:
        output = inputs.original + inputs.scaled
        return ProveResult(output, inputs.scaled, transcript)

    def verify(
        self, inputs: _MixInput, proof: Array, transcript: Transcript
    ) -> VerifyResult[Array]:
        output = inputs.original + inputs.scaled
        return VerifyResult(output, transcript, proof == inputs.scaled)


@dataclass(frozen=True)
class _CompositeProof:
    scale: Array
    mix: Array


class _CompositeStage(Stage[Array, Array, Array, Array, _CompositeProof]):
    """Explicit dataflow keeps the original input live across the first child."""

    name = "composite"

    def __init__(self) -> None:
        self.scale = _ScaleStage(3)
        self.mix = _MixStage()

    def prove(
        self, inputs: Array, transcript: Transcript
    ) -> ProveResult[Array, _CompositeProof]:
        scaled = self.scale.prove(inputs, transcript)
        mixed = self.mix.prove(
            _MixInput(original=inputs, scaled=scaled.output), scaled.transcript
        )
        return ProveResult(
            mixed.output,
            _CompositeProof(scale=scaled.proof, mix=mixed.proof),
            mixed.transcript,
        )

    def verify(
        self,
        inputs: Array,
        proof: _CompositeProof,
        transcript: Transcript,
    ) -> VerifyResult[Array]:
        scaled = self.scale.verify(inputs, proof.scale, transcript)
        mixed = self.mix.verify(
            _MixInput(original=inputs, scaled=scaled.output),
            proof.mix,
            scaled.transcript,
        )
        return VerifyResult(mixed.output, mixed.transcript, scaled.ok & mixed.ok)


class StageTest(absltest.TestCase):
    def test_composite_threads_named_dependencies(self) -> None:
        protocol = _CompositeStage()
        initial = fnp.array(11, KB)

        proved = protocol.prove(initial, cheap_transcript(KB))
        verified = protocol.verify(initial, proved.proof, cheap_transcript(KB))

        self.assertTrue(bool(verified.ok))
        self.assertTrue(bool(verified.output == proved.output))

    def test_composite_rejects_tampered_child_proof(self) -> None:
        protocol = _CompositeStage()
        initial = fnp.array(9, KB)
        proved = protocol.prove(initial, cheap_transcript(KB))
        bad = replace(
            proved.proof,
            mix=proved.proof.mix + fnp.array(1, KB),
        )

        verified = protocol.verify(initial, bad, cheap_transcript(KB))
        self.assertFalse(bool(verified.ok))


if __name__ == "__main__":
    absltest.main()
