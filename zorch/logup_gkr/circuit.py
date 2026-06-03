# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR fractional-sum circuit.

A layer is four equal-length MLEs `(n0, n1, d0, d1)`: index `i` carries two
binary-tree children, the fractions `n0[i]/d0[i]` and `n1[i]/d1[i]`. A
transition folds each pair into one fraction --

    n0/d0 + n1/d1 = (n0*d1 + n1*d0) / (d0*d1)

-- pairing LSB-adjacent nodes (stride 2), which halves every MLE and eliminates
one (row) variable. Iterating to the interaction floor (`num_row_variables == 0`)
leaves one fraction per interaction; `extract_outputs` interleaves the two
children back into the output numerator/denominator MLEs over the interaction
variables plus one.

Dense/uniform layout only: every interaction shares one row count, so a layer is
a flat power of two with no padding. Jagged real-chip layouts (per-interaction
heights, gather-pad) and interaction fingerprinting are an SP1-trace concern and
live in the consumer (whir-zorch), not here -- zorch stays scheme-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class GkrLayer:
    """One dense fractional-sum layer over (interaction || row) variables.

    `num_interaction_variables` is the floor: folding stops once the row
    variables are exhausted and only the interaction dimension remains.
    """

    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array
    num_interaction_variables: int

    def __post_init__(self) -> None:
        # Reject malformed layers at construction rather than at a later
        # broadcast or negative-row-count. num_variables also checks the
        # power-of-two width via log2_strict_usize.
        shape = self.numerator_0.shape
        for name in ("numerator_1", "denominator_0", "denominator_1"):
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"all MLEs must share a shape; {name} is "
                    f"{getattr(self, name).shape}, numerator_0 is {shape}"
                )
        if not 0 <= self.num_interaction_variables <= self.num_variables:
            raise ValueError(
                f"num_interaction_variables must be in [0, {self.num_variables}], "
                f"got {self.num_interaction_variables}"
            )

    @property
    def num_variables(self) -> int:
        return log2_strict_usize(self.numerator_0.shape[0])

    @property
    def num_row_variables(self) -> int:
        return self.num_variables - self.num_interaction_variables


@dataclass(frozen=True)
class LogUpGkrOutput:
    """Final numerator/denominator MLEs after all transitions."""

    numerator: Array
    denominator: Array


def layer_transition(layer: GkrLayer) -> GkrLayer:
    """Fold one row variable: sum each LSB-adjacent fraction pair (stride 2).

    Even nodes become the next layer's 0-child, odd nodes its 1-child. Halves
    every MLE. Requires a row variable to fold.
    """
    if layer.num_row_variables < 1:
        raise ValueError("no row variable left to fold")
    n0e, n0o = layer.numerator_0[0::2], layer.numerator_0[1::2]
    n1e, n1o = layer.numerator_1[0::2], layer.numerator_1[1::2]
    d0e, d0o = layer.denominator_0[0::2], layer.denominator_0[1::2]
    d1e, d1o = layer.denominator_1[0::2], layer.denominator_1[1::2]
    return GkrLayer(
        numerator_0=n0e * d1e + n1e * d0e,
        numerator_1=n0o * d1o + n1o * d0o,
        denominator_0=d0e * d1e,
        denominator_1=d0o * d1o,
        num_interaction_variables=layer.num_interaction_variables,
    )


def build_pyramid(first: GkrLayer) -> list[GkrLayer]:
    """Eager fold from `first` (most row variables) down to the interaction floor.

    Eager rather than one fused program: the full pyramid does not fit one
    `@jit` at scale, and each transition's output feeds the next layer's
    per-variable sumcheck independently.
    """
    layers = [first]
    while layers[-1].num_row_variables > 0:
        layers.append(layer_transition(layers[-1]))
    return layers


def _interleave(child_0: Array, child_1: Array) -> Array:
    """Interleave two equal-length children into [c0[0], c1[0], c0[1], c1[1], ...]."""
    return jnp.stack([child_0, child_1], axis=-1).flatten()


def extract_outputs(layer: GkrLayer) -> LogUpGkrOutput:
    """Interleave the two children of the floor layer into the output MLEs.

    At `num_row_variables == 0` each index is one interaction's fraction with a
    0-child and a 1-child; interleaving recovers the MLE over the interaction
    variables plus one.
    """
    if layer.num_row_variables != 0:
        raise ValueError(
            f"extract_outputs expects the interaction floor (num_row_variables "
            f"== 0), got {layer.num_row_variables}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )
