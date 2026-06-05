# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KZG structured reference string, split into the prover and verifier keys.

A single trusted setup fixes a secret `τ` and derives both keys from it. They are
stored apart because their sizes and lifetimes differ: the proving key is the full
`τⁱ` ladder in G1 (O(degree), megabytes) and the verifier key is three fixed group
elements (O(1)). The verifier deployment ships only `KzgVerifierKey`. That
`pk.powers_g1` and `vk.tau_g2` come from the *same* `τ` is the soundness invariant
— the analog of the sumcheck block's shared, module-level oracle expression that
keeps prover and verifier from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class KzgProvingKey:
    """O(degree). `powers_g1[i] = τⁱ · G1` (bn254 G1 affine [N]). Never shipped to
    a verifier."""

    powers_g1: Array


@dataclass(frozen=True)
class KzgVerifierKey:
    """O(1), degree-independent — the whole of what a verifier deployment ships."""

    gen_g1: Array  # bn254 G1 affine — [1]₁
    gen_g2: Array  # bn254 G2 affine — [1]₂
    tau_g2: Array  # bn254 G2 affine — [τ]₂


def setup(
    powers_g1: Array, tau_g2: Array, gen_g1: Array, gen_g2: Array
) -> tuple[KzgProvingKey, KzgVerifierKey]:
    """Split one SRS (all derived from the same `τ`) into `(proving_key,
    verifier_key)`."""
    return KzgProvingKey(powers_g1), KzgVerifierKey(gen_g1, gen_g2, tau_g2)
