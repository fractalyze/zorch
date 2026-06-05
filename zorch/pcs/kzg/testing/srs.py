# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Toy KZG SRS fixture: scalar-mul the bn254 generators by powers of a *known* τ.
Tests only — a real SRS comes from a trusted-setup ceremony where τ is destroyed.

Standard-domain bn254 dtypes, deliberately: `lax.pairing_check` mis-decodes
Montgomery-form inputs (fractalyze/zkx#518), so a mont KZG round trip cannot
verify yet. Switch to the `*_mont` test convention when that lands — the
scalar-muls and `lax.msm` are already mont-correct."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import zk_dtypes
from jax import lax

from zorch.pcs.kzg.setup import KzgProvingKey, KzgVerifierKey, setup

SF = zk_dtypes.bn254_sf
G1 = zk_dtypes.bn254_g1_affine
G2 = zk_dtypes.bn254_g2_affine


def toy_srs(tau: int, n: int) -> tuple[KzgProvingKey, KzgVerifierKey]:
    """An `n`-power SRS from a known τ (CPU/GPU EC mul). The generators and Fr
    modulus come from zk_dtypes; a round-trip over the keys certifies the SRS is
    internally consistent."""
    fr = zk_dtypes.pfinfo(SF).modulus
    e1, e2 = zk_dtypes.ecinfo(G1), zk_dtypes.ecinfo(G2)
    g1 = jnp.asarray(G1((e1.gx, e1.gy)))
    g2 = jnp.asarray(
        G2((tuple(cast("list[int]", e2.gx)), tuple(cast("list[int]", e2.gy))))
    )
    powers = [
        lax.convert_element_type(jnp.array(pow(tau, i, fr), SF) * g1, G1)
        for i in range(n)
    ]
    tau_g2 = lax.convert_element_type(jnp.array(tau, SF) * g2, G2)
    return setup(jnp.stack(powers), tau_g2, g1, g2)
