# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and chain runners for LogUp-GKR tests."""

from __future__ import annotations

import functools
from typing import Any

import frx
import frx.numpy as fnp
import zk_dtypes
from frx import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    JaggedGkrLayer,
    LogUpGkrOutput,
    build_pyramid,
    extract_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.prover import Carry, LayerProof, bind_output
from zorch.logup_gkr.prover import GkrLayerRound as _ProverLayer
from zorch.logup_gkr.verifier import GkrLayerRound as _VerifierLayer
from zorch.round import prove_rounds, verify_rounds
from zorch.sumcheck.jagged.types import RoundWidthCaps
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

_KB = zk_dtypes.koalabear_mont
_EF = zk_dtypes.koalabearx4_mont


def host_counts(layer: JaggedGkrLayer) -> tuple[int, ...]:
    """The layer's traced counts read back as host ints — the testkit oracle
    seam (an eager device read, so never for production code)."""
    return tuple(int(rc) for rc in layer.row_counts)


def random_jagged_layer(
    seed: int, row_counts: tuple[int, ...], dtype: Any = _KB
) -> JaggedGkrLayer:
    """A random jagged GKR layer over `row_counts` (flat, interaction-major),
    at the zero-slack width."""
    height = sum(row_counts)
    return JaggedGkrLayer(
        numerator_0=rand_field(seed, (height,), dtype),
        numerator_1=rand_field(seed + 1, (height,), dtype),
        denominator_0=rand_field(seed + 2, (height,), dtype),
        denominator_1=rand_field(seed + 3, (height,), dtype),
        row_counts=fnp.asarray(row_counts, fnp.int32),
    )


def widen_jagged_layer(layer: JaggedGkrLayer, width: int) -> JaggedGkrLayer:
    """The same layer at a wider capacity: live prefix copied, dead tail
    zero. Same counts, so downstream byte-equality across widths is exactly
    the capacity contract."""

    def lay(a: Array) -> Array:
        return fnp.concatenate([a, fnp.zeros((width - a.shape[0],), dtype=a.dtype)])

    return JaggedGkrLayer(
        numerator_0=lay(layer.numerator_0),
        numerator_1=lay(layer.numerator_1),
        denominator_0=lay(layer.denominator_0),
        denominator_1=lay(layer.denominator_1),
        row_counts=layer.row_counts,
    )


def mixed_field_jagged_layer(
    seed: int, row_counts: tuple[int, ...], base: Any = _KB, ext: Any = _EF
) -> JaggedGkrLayer:
    """A jagged layer with base-field numerators under extension-field
    denominators -- the first-layer shape when numerators are LogUp
    multiplicities (naturally base-field). Denominators are generic extensions."""
    height = sum(row_counts)
    return JaggedGkrLayer(
        numerator_0=rand_field(seed, (height,), base),
        numerator_1=rand_field(seed + 1, (height,), base),
        denominator_0=rand_ext_field(seed + 2, (height,), base, ext),
        denominator_1=rand_ext_field(seed + 3, (height,), base, ext),
        row_counts=fnp.asarray(row_counts, fnp.int32),
    )


def virtual_planes(
    layer: JaggedGkrLayer, num_row_variables: int
) -> tuple[Array, Array, Array, Array]:
    """The four planes as virtual dense MLEs over (interaction || row)
    variables -- each segment zero/one-extended to `2^num_row_variables` rows
    with the fold-neutral fraction (n=0, d=1). The brute-force oracle the
    jagged prover's closed-form corrections are checked against."""
    counts = host_counts(layer)
    starts = [0]
    for rc in counts:
        starts.append(starts[-1] + rc)
    rows = 1 << num_row_variables
    planes = []
    for arr, neutral in (
        (layer.numerator_0, 0),
        (layer.numerator_1, 0),
        (layer.denominator_0, 1),
        (layer.denominator_1, 1),
    ):
        parts = []
        for i, rc in enumerate(counts):
            pad = (
                fnp.zeros((rows - rc,), arr.dtype)
                if neutral == 0
                else fnp.ones((rows - rc,), arr.dtype)
            )
            parts.append(fnp.concatenate([arr[starts[i] : starts[i + 1]], pad]))
        planes.append(fnp.concatenate(parts))
    return planes[0], planes[1], planes[2], planes[3]


def build_jagged_pyramid(first: JaggedGkrLayer) -> list[JaggedGkrLayer]:
    """Fold to the floor under a schedule that over-pads unsaturated segments
    to even counts -- saturated segments stay at one row so the floor is
    reachable while the padding paths stay exercised."""
    layers = [first]
    while max(host_counts(layers[-1])) > 1:
        folded = tuple((rc + 1) // 2 for rc in host_counts(layers[-1]))
        schedule = tuple(fc if fc == 1 else fc + fc % 2 for fc in folded)
        layers.append(jagged_layer_transition(layers[-1], schedule))
    return layers


def caps_for(row_counts: tuple[int, ...], num_row_variables: int) -> RoundWidthCaps:
    """Tight width caps for a test layout: the layer's own widths rounded up
    only where the ABI demands (elements to a multiple of 4, interaction to
    >= 4). Rounds are caps-mandatory, so every test chain needs one."""
    width = sum(rc + rc % 2 for rc in row_counts)
    return RoundWidthCaps(
        elements=width + (-width % 4),
        eq_row=1 << num_row_variables,
        interaction=max(4, len(row_counts)),
    )


def random_first_layer(
    seed: int, num_batch_variables: int, num_row_variables: int
) -> GkrLayer:
    """A random dense first GKR layer, 2^(int+row) wide per MLE. Montgomery field
    -- it matches the poseidon2 sponge and the prove's production field; the
    standard domain is too slow to be worth a second representation."""
    width = 1 << (num_batch_variables + num_row_variables)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), _KB),
        numerator_1=rand_field(seed + 1, (width,), _KB),
        denominator_0=rand_field(seed + 2, (width,), _KB),
        denominator_1=rand_field(seed + 3, (width,), _KB),
        num_batch_variables=num_batch_variables,
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
    final, _, proofs = prove_rounds(
        [_ProverLayer(layer) for layer in reversed(layers[:-1])], carry, transcript
    )
    return layers, output, proofs, final


def prove_gkr(
    first: GkrLayer,
) -> tuple[list[GkrLayer], LogUpGkrOutput, list[LayerProof], Carry]:
    """`prove_gkr_with_transcript` over a cheap deterministic test transcript."""
    return prove_gkr_with_transcript(first, cheap_transcript(_KB))


@functools.partial(frx.jit, static_argnums=(4,))
def prove_gkr_jitted(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    num_batch_variables: int,
) -> Array:
    """`prove_gkr` traced into one fused program (whole pyramid + chain).

    The four first-layer MLEs are the traced inputs; `num_batch_variables`
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
        num_batch_variables=num_batch_variables,
    )
    return prove_gkr(first)[2][-1].round_polys


def verify_gkr_with_transcript(
    output: LogUpGkrOutput, proofs: list[LayerProof], transcript: Transcript
) -> tuple[Carry, Array]:
    """Run the GKR verifier chain, re-deriving challenges from `transcript` (the
    dual of `prove_gkr_with_transcript`). Returns (final_carry, ok)."""
    carry, transcript = bind_output(output, transcript)
    final, _, ok = verify_rounds(
        [_VerifierLayer() for _ in proofs], carry, proofs, transcript
    )
    return final, ok


def verify_gkr(output: LogUpGkrOutput, proofs: list[LayerProof]) -> tuple[Carry, Array]:
    """`verify_gkr_with_transcript` over a cheap deterministic test transcript."""
    return verify_gkr_with_transcript(output, proofs, cheap_transcript(_KB))
