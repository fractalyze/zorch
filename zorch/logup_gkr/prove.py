# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR prover.

Builds the fractional-sum pyramid (`circuit.build_pyramid`), then proves one
sumcheck per layer from the interaction floor outward to the input. Each layer's
sumcheck is the agnostic `LogupGkrRound` driven by the generic `prove` folder;
the cross-layer glue here is the GKR-specific part: sample the batching challenge
lambda, observe the four openings, sample the child-selector `r`, and grow the
evaluation point by binding `r` as the next layer's LSB.

Layers are folded and proved eagerly (a Python loop over `reversed`), not one
fused program: the pyramid does not fit one `@jit` at scale, and each layer's
sumcheck is independent given the running evaluation point.

Scheme-agnostic: no interaction model, jagged layout, or trace openings -- those
are the consumer's (whir-zorch) job. The proof carries only the circuit output
and, per layer, the round polynomials plus the four scalar openings.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr.circuit import (
    GkrLayer,
    LogUpGkrOutput,
    build_pyramid,
    extract_outputs,
)
from zorch.logup_gkr.round import LogupGkrRound
from zorch.prove import prove
from zorch.transcript import Transcript


@dataclass(frozen=True)
class LayerProof:
    """One GKR layer's sumcheck transcript: round polynomials + final openings."""

    round_polys: Array  # (num_variables, degree + 1), each round's univariate
    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array


@dataclass(frozen=True)
class LogupGkrProof:
    output: LogUpGkrOutput
    layer_proofs: list[LayerProof]


def _eq_weights(point: Array) -> Array:
    """Tensor-product eq weights over the hypercube.

    `weights[x] = prod_j eq(point[j], bit_j(x))`, with `point[j]` bound to bit
    `j` of the flat index (`point[0]` least-significant). The round folds
    MSB-first, so callers flip the recovered challenges before pairing them with
    this layout. Built by `range` indexing rather than iterating the array
    (iterating an extension-field array dispatches `lax.sign` on the GPU
    backend).
    """
    one = jnp.ones((), dtype=point.dtype)
    weights = jnp.ones((1,), dtype=point.dtype)
    for j in range(point.shape[0]):
        z = point[j]
        weights = jnp.concatenate([weights * (one - z), weights * z])
    return weights


def _recover_challenges(transcript: Transcript, round_polys: Array) -> Array:
    """Re-derive a layer's sumcheck challenges from a fork of the transcript.

    `prove` consumes the challenges internally; the cross-layer point needs
    them. Fiat-Shamir is deterministic and the transcript is immutable, so
    replaying observe-then-sample over the round polynomials from the
    pre-sumcheck state reproduces exactly the challenges the rounds used,
    without disturbing the live transcript.
    """
    chals = []
    for i in range(round_polys.shape[0]):
        transcript = transcript.observe(round_polys[i])
        transcript, c = transcript.sample(1)
        chals.append(c[0])
    return jnp.stack(chals)


def prove_logup_gkr(
    first: GkrLayer, transcript: Transcript
) -> tuple[Transcript, LogupGkrProof]:
    """Prove the dense LogUp-GKR circuit grown from `first`."""
    layers = build_pyramid(first)
    output = extract_outputs(layers[-1])
    initial_num_vars = layers[-1].num_interaction_variables + 1

    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(output.denominator)
    transcript, eval_point = transcript.sample(initial_num_vars)

    layer_proofs: list[LayerProof] = []
    # Interaction floor outward to the input; each step binds one more variable.
    for layer in reversed(layers[:-1]):
        transcript, lam = transcript.sample(1)
        # State order is LogupGkrRound's: [eq, n0, d1, n1, d0].
        state = [
            _eq_weights(eval_point),
            layer.numerator_0,
            layer.denominator_1,
            layer.numerator_1,
            layer.denominator_0,
        ]
        round = LogupGkrRound(lam[0])
        pre = transcript
        final_state, transcript, round_polys = prove(round, state, transcript)
        challenges = _recover_challenges(pre, round_polys)

        _, n0, d1, n1, d0 = (factor[0] for factor in final_state)
        transcript = transcript.observe(jnp.stack([n0, n1, d0, d1]))
        transcript, r = transcript.sample(1)
        # The round binds MSB-first (SumcheckRound splits into halves), so the
        # j-th challenge binds bit (n-1-j); flip to index-order before binding
        # the child selector r as the next (bigger) layer's LSB.
        eval_point = jnp.concatenate([r, jnp.flip(challenges)])

        layer_proofs.append(
            LayerProof(
                round_polys=round_polys,
                numerator_0=n0,
                numerator_1=n1,
                denominator_0=d0,
                denominator_1=d1,
            )
        )

    return transcript, LogupGkrProof(output=output, layer_proofs=layer_proofs)
