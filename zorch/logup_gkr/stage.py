# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR role implementations.

The claim types and the proof envelope are layout-independent, so the jagged
roles (`jagged_stage`) reduce the same `LogUpOutputClaim` to the same
`InputLayerClaim` and a consumer can swap one layout for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import GkrLayer, LogUpGkrOutput, build_pyramid
from zorch.logup_gkr.prover import GkrLayerRound as ProverLayerRound
from zorch.logup_gkr.prover import LayerClaim, LayerProof, bind_output
from zorch.logup_gkr.verifier import GkrLayerRound as VerifierLayerRound
from zorch.round import prove_rounds, verify_rounds
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.transcript import Transcript

LayerProofT = TypeVar("LayerProofT")


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
class GkrProof(Generic[LayerProofT]):
    """One conditional reduction-proof section per GKR layer.

    Parameterized by the layer proof because the pyramid's shape is the
    protocol and the layer's sumcheck transcript is the layout.
    """

    layers: tuple[LayerProofT, ...]


def _input_claim(claim: LayerClaim) -> InputLayerClaim:
    numerator, denominator, point = claim
    return InputLayerClaim(numerator, denominator, point)


class LogUpGkrProver(
    ProverStage[LogUpOutputClaim, GkrLayer, InputLayerClaim, GkrProof[LayerProof]]
):
    """Prove an output claim conditional on an input-layer claim."""

    def __init__(self, challenges: ChallengePolicy) -> None:
        self.challenges = challenges

    def prove(
        self,
        claim: LogUpOutputClaim,
        witness: GkrLayer,
        transcript: Transcript,
    ) -> ProveResult[InputLayerClaim, GkrProof[LayerProof]]:
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


class LogUpGkrVerifier(
    VerifierStage[LogUpOutputClaim, InputLayerClaim, GkrProof[LayerProof]]
):
    """Verify an output-to-input-layer LogUp-GKR reduction."""

    def __init__(self, challenges: ChallengePolicy) -> None:
        self.challenges = challenges

    def verify(
        self,
        claim: LogUpOutputClaim,
        reduction_proof: GkrProof[LayerProof],
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


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _p: type[
        ProverStage[(LogUpOutputClaim, GkrLayer, InputLayerClaim, GkrProof[LayerProof])]
    ] = LogUpGkrProver
    _v: type[
        VerifierStage[(LogUpOutputClaim, InputLayerClaim, GkrProof[LayerProof])]
    ] = LogUpGkrVerifier
