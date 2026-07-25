# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR role implementations."""

from __future__ import annotations

from dataclasses import dataclass

from frx import Array

from zorch.challenge import DEFAULT_CHALLENGES, ChallengePolicy
from zorch.logup_gkr.circuit import GkrLayer, LogUpGkrOutput, build_pyramid
from zorch.logup_gkr.prover import Carry, LayerProof, bind_output
from zorch.logup_gkr.prover import GkrLayerRound as ProverLayerRound
from zorch.logup_gkr.verifier import GkrLayerRound as VerifierLayerRound
from zorch.round import prove_rounds, verify_rounds
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class LogUpOutputClaim:
    """Public LogUp circuit-output claim and verifier-owned layer count."""

    output: LogUpGkrOutput
    layers: int


@dataclass(frozen=True)
class InputLayerClaim:
    """Numerator and denominator evaluations claimed at the input layer."""

    numerator: Array
    denominator: Array
    point: Array


@dataclass(frozen=True)
class GkrProof:
    """One conditional reduction-proof section per GKR layer."""

    layers: tuple[LayerProof, ...]


def _input_claim(carry: Carry) -> InputLayerClaim:
    numerator, denominator, point = carry
    return InputLayerClaim(numerator, denominator, point)


class LogUpGkrProver(
    ProverStage[LogUpOutputClaim, GkrLayer, InputLayerClaim, GkrProof]
):
    """Prove an output claim conditional on an input-layer claim."""

    def __init__(self, challenges: ChallengePolicy = DEFAULT_CHALLENGES) -> None:
        self.challenges = challenges

    def prove(
        self,
        claim: LogUpOutputClaim,
        witness: GkrLayer,
        transcript: Transcript,
    ) -> ProveResult[InputLayerClaim, GkrProof]:
        if witness.num_row_variables != claim.layers:
            raise ValueError(
                f"claim expects {claim.layers} GKR layers, "
                f"witness needs {witness.num_row_variables}"
            )
        pyramid = build_pyramid(witness)
        carry, transcript = bind_output(claim.output, transcript, self.challenges)
        carry, transcript, proofs = prove_rounds(
            (
                ProverLayerRound(layer, self.challenges)
                for layer in reversed(pyramid[:-1])
            ),
            carry,
            transcript,
        )
        reduction_proofs = tuple(proofs)
        if len(reduction_proofs) != claim.layers:
            raise AssertionError(
                f"built {len(reduction_proofs)} GKR proofs for {claim.layers} layers"
            )
        return ProveResult(
            _input_claim(carry),
            GkrProof(reduction_proofs),
            transcript,
        )


class LogUpGkrVerifier(VerifierStage[LogUpOutputClaim, InputLayerClaim, GkrProof]):
    """Verify an output-to-input-layer LogUp-GKR reduction."""

    def __init__(self, challenges: ChallengePolicy = DEFAULT_CHALLENGES) -> None:
        self.challenges = challenges

    def verify(
        self,
        claim: LogUpOutputClaim,
        reduction_proof: GkrProof,
        transcript: Transcript,
    ) -> VerifyResult[InputLayerClaim]:
        if len(reduction_proof.layers) != claim.layers:
            raise ValueError(
                f"expected {claim.layers} GKR layers, "
                f"got {len(reduction_proof.layers)}"
            )
        carry, transcript = bind_output(claim.output, transcript, self.challenges)
        carry, transcript, ok = verify_rounds(
            (VerifierLayerRound(self.challenges) for _ in range(claim.layers)),
            carry,
            reduction_proof.layers,
            transcript,
        )
        return VerifyResult(_input_claim(carry), transcript, ok)
