# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged LogUp-GKR role implementations.

The same reduction the dense roles run -- a circuit-output claim down to an
input-layer point claim -- over the jagged layout. Claim types come from
`stage`, so the two layouts are interchangeable at the stage seam and only
the witness and the layer proofs differ.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import JaggedGkrLayer, build_jagged_pyramid
from zorch.logup_gkr.jagged_prover import JaggedGkrLayerRound as ProverLayerRound
from zorch.logup_gkr.jagged_prover import JaggedLayerProof
from zorch.logup_gkr.jagged_verifier import JaggedGkrLayerRound as VerifierLayerRound
from zorch.logup_gkr.prover import bind_output
from zorch.logup_gkr.stage import (
    GkrProof,
    InputLayerClaim,
    LogUpOutputClaim,
    _input_claim,
)
from zorch.round import prove_rounds, verify_rounds
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.sumcheck.jagged.buffers import LayerBuffers
from zorch.sumcheck.jagged.types import RoundWidthCaps
from zorch.transcript import Transcript


@dataclass(frozen=True)
class JaggedGkrWitness:
    """The input jagged layer and the per-transition fold schedule carrying it
    to the batch floor.

    The schedule follows the row counts, so it is per-input and rides the
    witness; the width caps are a capacity class shared across inputs and ride
    the prover instead. `schedules[k]` is the argument
    `circuit.jagged_layer_transition` takes -- `(out_row_counts, out_width)`
    traced, or a bare host sequence at its zero-slack width.
    """

    input_layer: JaggedGkrLayer
    schedules: Sequence[tuple[Array, int] | Sequence[int]]


class JaggedLogUpGkrProver(
    ProverStage[
        LogUpOutputClaim,
        JaggedGkrWitness,
        InputLayerClaim,
        GkrProof[JaggedLayerProof],
    ]
):
    """Prove an output claim conditional on an input-layer claim, jagged."""

    def __init__(
        self,
        caps: RoundWidthCaps,
        challenges: ChallengePolicy,
    ) -> None:
        self.caps = caps
        self.challenges = challenges

    def prove(
        self,
        claim: LogUpOutputClaim,
        witness: JaggedGkrWitness,
        transcript: Transcript,
    ) -> ProveResult[InputLayerClaim, GkrProof[JaggedLayerProof]]:
        if len(witness.schedules) != claim.layers:
            raise ValueError(
                f"claim expects {claim.layers} GKR layers, "
                f"witness folds through {len(witness.schedules)}"
            )
        pyramid = build_jagged_pyramid(witness.input_layer, witness.schedules)
        # The floor holds the public output; the chain proves the rest.
        pyramid.pop()
        carry, transcript = bind_output(claim.output, transcript, self.challenges)
        # One `LayerBuffers` per chain: the cap-wide planes are ~GiB per class,
        # so the holder must die with the prove. Layers leave the list as the
        # chain consumes them, keeping one intermediate layer resident instead
        # of the whole pyramid.
        buffers = LayerBuffers()

        def rounds() -> Iterator[ProverLayerRound]:
            while pyramid:
                yield ProverLayerRound(
                    pyramid.pop(),
                    self.challenges,
                    caps=self.caps,
                    layer_bufs=buffers,
                )

        carry, transcript, proofs = prove_rounds(rounds(), carry, transcript)
        return ProveResult(
            _input_claim(carry),
            GkrProof(tuple(proofs)),
            transcript,
        )


class JaggedLogUpGkrVerifier(
    VerifierStage[LogUpOutputClaim, InputLayerClaim, GkrProof[JaggedLayerProof]]
):
    """Verify an output-to-input-layer LogUp-GKR reduction, jagged.

    Layout-blind, exactly like its rounds: the prover's virtual-mass
    corrections make the round polynomials those of the virtual dense
    hypercube, so nothing here reads a row count.
    """

    def __init__(self, challenges: ChallengePolicy) -> None:
        self.challenges = challenges

    def verify(
        self,
        claim: LogUpOutputClaim,
        reduction_proof: GkrProof[JaggedLayerProof],
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
    _p: type[
        ProverStage[
            (
                LogUpOutputClaim,
                JaggedGkrWitness,
                InputLayerClaim,
                GkrProof[JaggedLayerProof],
            )
        ]
    ] = JaggedLogUpGkrProver
    _v: type[
        VerifierStage[(LogUpOutputClaim, InputLayerClaim, GkrProof[JaggedLayerProof])]
    ] = JaggedLogUpGkrVerifier
