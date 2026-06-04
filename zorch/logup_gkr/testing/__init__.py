# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and chain runners for LogUp-GKR tests."""

from __future__ import annotations

import functools

import jax
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
from zorch.transcript import StubTranscript, Transcript

_KB = zk_dtypes.koalabear_mont


def random_first_layer(
    seed: int, num_interaction_variables: int, num_row_variables: int
) -> GkrLayer:
    """A random dense first GKR layer, 2^(int+row) wide per MLE. Montgomery field
    -- it matches the poseidon2 sponge and the prove's production field; the
    standard domain is too slow to be worth a second representation."""
    width = 1 << (num_interaction_variables + num_row_variables)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), _KB),
        numerator_1=rand_field(seed + 1, (width,), _KB),
        denominator_0=rand_field(seed + 2, (width,), _KB),
        denominator_1=rand_field(seed + 3, (width,), _KB),
        num_interaction_variables=num_interaction_variables,
    )


def prove_gkr_with_transcript(
    first: GkrLayer, transcript: Transcript
) -> tuple[list[GkrLayer], LogUpGkrOutput, list[LayerProof], Carry]:
    """Run the GKR prover chain over `first`'s pyramid, drawing Fiat-Shamir
    challenges from `transcript` (a preset `StubTranscript` or the on-device
    poseidon2 `DuplexTranscript`).

    Returns (layers, output, layer_proofs, final_carry).
    """
    layers = build_pyramid(first)
    output = extract_outputs(layers[-1])
    carry, transcript = bind_output(output, transcript)
    chain = ProveChain([_ProverLayer(layer) for layer in reversed(layers[:-1])])
    final, _, proofs = chain(carry, transcript)
    return layers, output, proofs, final


def prove_gkr(
    first: GkrLayer, challenges: Array
) -> tuple[list[GkrLayer], LogUpGkrOutput, list[LayerProof], Carry]:
    """`prove_gkr_with_transcript` against a preset challenge stream."""
    return prove_gkr_with_transcript(first, StubTranscript(challenges))


@functools.partial(jax.jit, static_argnums=(5,))
def prove_gkr_jitted(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    challenges: Array,
    num_interaction_variables: int,
) -> Array:
    """`prove_gkr` traced into one fused program (whole pyramid + chain).

    The four first-layer MLEs and the challenge stream are the traced inputs;
    `num_interaction_variables` is static (it fixes the pyramid height, hence the
    unrolled layer count). Returns the last proved layer's round polynomials --
    the tail of the sequential carry, so it transitively forces the whole prove.
    Drives the jit-vs-eager equivalence test, which must fuse the *same* program.
    """
    first = GkrLayer(
        numerator_0,
        numerator_1,
        denominator_0,
        denominator_1,
        num_interaction_variables=num_interaction_variables,
    )
    return prove_gkr(first, challenges)[2][-1].round_polys


def verify_gkr_with_transcript(
    output: LogUpGkrOutput, proofs: list[LayerProof], transcript: Transcript
) -> tuple[Carry, Array]:
    """Run the GKR verifier chain, re-deriving challenges from `transcript` (the
    dual of `prove_gkr_with_transcript`). Returns (final_carry, ok)."""
    carry, transcript = bind_output(output, transcript)
    chain = VerifyChain([_VerifierLayer() for _ in proofs])
    final, _, ok = chain(carry, proofs, transcript)
    return final, ok


def verify_gkr(
    output: LogUpGkrOutput, proofs: list[LayerProof], challenges: Array
) -> tuple[Carry, Array]:
    """`verify_gkr_with_transcript` against a preset challenge stream."""
    return verify_gkr_with_transcript(output, proofs, StubTranscript(challenges))
