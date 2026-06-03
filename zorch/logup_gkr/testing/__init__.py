# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and chain runners for LogUp-GKR tests."""

from __future__ import annotations

import zk_dtypes
from jax import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    LogUpGkrOutput,
    build_pyramid,
    extract_outputs,
)
from zorch.logup_gkr.prover import Carry, LayerProof, bind_output
from zorch.logup_gkr.prover import GkrLayerRound as _ProverLayer
from zorch.logup_gkr.verifier import GkrLayerRound as _VerifierLayer
from zorch.round import ProveChain, VerifyChain
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

_KB = zk_dtypes.koalabear


def random_first_layer(
    seed: int, num_interaction_variables: int, num_row_variables: int
) -> GkrLayer:
    """A random dense first GKR layer, 2^(int+row) wide per MLE."""
    width = 1 << (num_interaction_variables + num_row_variables)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), _KB),
        numerator_1=rand_field(seed + 1, (width,), _KB),
        denominator_0=rand_field(seed + 2, (width,), _KB),
        denominator_1=rand_field(seed + 3, (width,), _KB),
        num_interaction_variables=num_interaction_variables,
    )


def prove_gkr(
    first: GkrLayer, challenges: Array
) -> tuple[list[GkrLayer], LogUpGkrOutput, list[LayerProof], Carry]:
    """Run the GKR prover chain over `first`'s pyramid.

    Returns (layers, output, layer_proofs, final_carry).
    """
    layers = build_pyramid(first)
    output = extract_outputs(layers[-1])
    carry, transcript = bind_output(output, StubTranscript(challenges))
    chain = ProveChain([_ProverLayer(layer) for layer in reversed(layers[:-1])])
    final, _, proofs = chain(carry, transcript)
    return layers, output, proofs, final


def verify_gkr(
    output: LogUpGkrOutput, proofs: list[LayerProof], challenges: Array
) -> tuple[Carry, Array]:
    """Run the GKR verifier chain. Returns (final_carry, ok)."""
    carry, transcript = bind_output(output, StubTranscript(challenges))
    chain = VerifyChain([_VerifierLayer() for _ in proofs])
    final, _, ok = chain(carry, proofs, transcript)
    return final, ok
