# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged LogUp-GKR role implementations.

The same reduction the dense roles run -- a circuit-output claim down to an
input-layer point claim -- over the jagged layout. Claim types come from
`stage`, so the two layouts are interchangeable at the stage seam and only
the witness and the layer proofs differ.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
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


def _caps_from_widths(widths: list[int], flat: int) -> tuple[int, ...] | None:
    """Each layer's own width as its round-buffer cap, or `None` to keep the
    flat cap.

    Free where it applies: a layer already sized to a width the rounds accept
    needs no lay-in, so the only question is whether that width is admissible.
    It must be a multiple of 4 (the boundary handoff pairs the row-width state
    through two stride-2 halvings) and within the class's own cap. A layer's
    natural fold width is routinely neither -- the floor-adjacent layers land
    on 5, 10, 18 -- so this declines rather than rounds: rounding up would put
    the cap above the width the layer was built at and hand the lay-in right
    back.

    Whether this holds is the consumer's capacity decision, not an input
    property: widths that track an input's row counts key the round zone per
    shard. `element_ladder=` is how a consumer states class-fixed widths and
    has the pyramid built at them.
    """
    if not widths or widths[0] > flat:
        return None
    if any(w % 4 for w in widths):
        return None
    if any(a < b for a, b in zip(widths, widths[1:], strict=False)):
        return None
    return tuple(widths)


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
        *,
        element_ladder: Sequence[int] | None = None,
    ) -> None:
        self.caps = caps
        self.challenges = challenges
        self.element_ladder = None if element_ladder is None else tuple(element_ladder)

    def _layer_caps(self, depth: int) -> tuple[int, ...] | None:
        """This class's per-layer element caps, `None` for the flat-cap path.

        A ladder makes each proved layer arrive at the width its own round
        zone runs, which removes the per-layer lay-in dispatch. It has to be
        checked against the class rather than the input: `caps.elements`
        bounds layer 0, and every entry must not exceed it, or a layer would
        claim more round-buffer width than the class admits.
        """
        ladder = self.element_ladder
        if ladder is None:
            return None
        if len(ladder) != depth:
            raise ValueError(
                f"element_ladder must carry one cap per proved layer "
                f"({depth}), got {len(ladder)}"
            )
        if ladder[0] != self.caps.elements:
            raise ValueError(
                f"element_ladder[0] ({ladder[0]}) must be the class's own "
                f"element cap ({self.caps.elements}) -- layer 0 is unfolded"
            )
        if any(a < b for a, b in zip(ladder, ladder[1:], strict=False)):
            raise ValueError(f"element_ladder must be non-increasing, got {ladder}")
        return ladder

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
        ladder = self._layer_caps(len(witness.schedules))
        pyramid = build_jagged_pyramid(
            witness.input_layer,
            witness.schedules,
            # The floor is not proved and `extract_jagged_outputs` demands it
            # at exactly `num_batches`, so the last transition keeps its own
            # width whatever the ladder says.
            None if ladder is None else [*(c for c in ladder[1:]), None],
        )
        # The floor holds the public output; the chain proves the rest.
        pyramid.pop()
        if ladder is None:
            ladder = _caps_from_widths(
                [layer.width for layer in pyramid], self.caps.elements
            )
        carry, transcript = bind_output(claim.output, transcript, self.challenges)
        # One `LayerBuffers` per chain: the cap-wide planes are ~GiB per class,
        # so the holder must die with the prove. Layers leave the list as the
        # chain consumes them, keeping one intermediate layer resident instead
        # of the whole pyramid. Under a ladder every layer already arrives at
        # its own cap, so the pool's lay-in degenerates to a passthrough.
        buffers = LayerBuffers()

        def rounds() -> Iterator[ProverLayerRound]:
            while pyramid:
                index = len(pyramid) - 1
                layer = pyramid.pop()
                yield ProverLayerRound(
                    layer,
                    self.challenges,
                    caps=(
                        self.caps
                        if ladder is None
                        else replace(self.caps, elements=ladder[index])
                    ),
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
