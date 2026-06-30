# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""IPA public parameters — one shared key, no prover/verifier asymmetry.

IPA is transparent (no trusted setup): the parameters are a fixed Pedersen basis
`G = (G_0, …, G_{n-1})` and one independent generator `U` that carries the inner
product, both sampled from public randomness ("nothing-up-my-sleeve"). Prover and
verifier hold the *same* `IpaKey` — the degenerate case of the seam's
prover-key/verifier-key split, as with `FriParams`.

The structural contrast with KZG worth stating: KZG's verifier key is O(1) (three
fixed group elements), but **IPA's verifier needs the full size-`n` basis `G`** —
its final check `G_final = ⟨s, G⟩` is a size-`n` MSM. That O(n) verifier cost is
exactly the debt the IPA accumulation scheme defers (see verifier.py
`reduce_opening` and the accumulation-zorch study note); it is a property of IPA,
not a shortcoming of this key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class IpaKey:
    """Public IPA parameters, identical on both sides.

    `basis` is the Pedersen basis `G` (bn254 G1 affine `[N]`); `u` is the inner
    product generator `U` (a single bn254 G1 affine point), independent of every
    `basis` element. A commitment binds polynomials of length up to `N`."""

    basis: Array  # bn254 G1 affine [N] — Pedersen basis G
    u: Array  # bn254 G1 affine — inner-product generator U


def setup(basis: Array, u: Array) -> IpaKey:
    """Assemble the shared key from a public basis `G` and generator `U`. A real
    deployment derives both from a nothing-up-my-sleeve hash-to-curve; a fixture
    basis lives in `testing/basis.py`."""
    return IpaKey(basis, u)
