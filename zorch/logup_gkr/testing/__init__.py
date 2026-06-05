# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and chain runners for LogUp-GKR tests."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    JaggedGkrLayer,
    LogUpGkrOutput,
    build_pyramid,
    extract_outputs,
)
from zorch.logup_gkr.prover import Carry, LayerProof, bind_output
from zorch.logup_gkr.prover import GkrLayerRound as _ProverLayer
from zorch.logup_gkr.verifier import GkrLayerRound as _VerifierLayer
from zorch.round import ProveChain, VerifyChain
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

_KB = zk_dtypes.koalabear_mont


def random_jagged_layer(
    seed: int, row_counts: tuple[int, ...], dtype: Any = _KB
) -> JaggedGkrLayer:
    """A random jagged GKR layer over `row_counts` (flat, interaction-major)."""
    height = sum(row_counts)
    return JaggedGkrLayer(
        numerator_0=rand_field(seed, (height,), dtype),
        numerator_1=rand_field(seed + 1, (height,), dtype),
        denominator_0=rand_field(seed + 2, (height,), dtype),
        denominator_1=rand_field(seed + 3, (height,), dtype),
        row_counts=row_counts,
    )


def virtual_planes(
    layer: JaggedGkrLayer, num_row_variables: int
) -> tuple[Array, Array, Array, Array]:
    """The four planes as virtual dense MLEs over (interaction || row)
    variables -- each segment zero/one-extended to `2^num_row_variables` rows
    with the fold-neutral fraction (n=0, d=1). The brute-force oracle the
    jagged prover's closed-form corrections are checked against."""
    starts = layer.start_indices
    rows = 1 << num_row_variables
    planes = []
    for arr, neutral in (
        (layer.numerator_0, 0),
        (layer.numerator_1, 0),
        (layer.denominator_0, 1),
        (layer.denominator_1, 1),
    ):
        parts = []
        for i, rc in enumerate(layer.row_counts):
            pad = (
                jnp.zeros((rows - rc,), arr.dtype)
                if neutral == 0
                else jnp.ones((rows - rc,), arr.dtype)
            )
            parts.append(jnp.concatenate([arr[starts[i] : starts[i + 1]], pad]))
        planes.append(jnp.concatenate(parts))
    return planes[0], planes[1], planes[2], planes[3]


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
    challenges from `transcript` (a cheap test `DuplexTranscript` or the
    on-device poseidon2 one).

    Returns (layers, output, layer_proofs, final_carry).
    """
    layers = build_pyramid(first)
    output = extract_outputs(layers[-1])
    carry, transcript = bind_output(output, transcript)
    chain = ProveChain([_ProverLayer(layer) for layer in reversed(layers[:-1])])
    final, _, proofs = chain(carry, transcript)
    return layers, output, proofs, final


def prove_gkr(
    first: GkrLayer,
) -> tuple[list[GkrLayer], LogUpGkrOutput, list[LayerProof], Carry]:
    """`prove_gkr_with_transcript` over a cheap deterministic test transcript."""
    return prove_gkr_with_transcript(first, cheap_transcript(_KB))


@functools.partial(jax.jit, static_argnums=(4,))
def prove_gkr_jitted(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    num_interaction_variables: int,
) -> Array:
    """`prove_gkr` traced into one fused program (whole pyramid + chain).

    The four first-layer MLEs are the traced inputs; `num_interaction_variables`
    is static (it fixes the pyramid height, hence the unrolled layer count).
    Returns the last proved layer's round polynomials -- the tail of the
    sequential carry, so it transitively forces the whole prove. Drives the
    jit-vs-eager equivalence test, which must fuse the *same* program.
    """
    first = GkrLayer(
        numerator_0,
        numerator_1,
        denominator_0,
        denominator_1,
        num_interaction_variables=num_interaction_variables,
    )
    return prove_gkr(first)[2][-1].round_polys


def verify_gkr_with_transcript(
    output: LogUpGkrOutput, proofs: list[LayerProof], transcript: Transcript
) -> tuple[Carry, Array]:
    """Run the GKR verifier chain, re-deriving challenges from `transcript` (the
    dual of `prove_gkr_with_transcript`). Returns (final_carry, ok)."""
    carry, transcript = bind_output(output, transcript)
    chain = VerifyChain([_VerifierLayer() for _ in proofs])
    final, _, ok = chain(carry, proofs, transcript)
    return final, ok


def verify_gkr(output: LogUpGkrOutput, proofs: list[LayerProof]) -> tuple[Carry, Array]:
    """`verify_gkr_with_transcript` over a cheap deterministic test transcript."""
    return verify_gkr_with_transcript(output, proofs, cheap_transcript(_KB))
