# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Toy IPA Pedersen basis fixture: scalar-mul the bn254 G1 generator by known,
distinct seeds. Tests only — a real basis is hash-to-curve from public
randomness, with no known discrete logs (which here would let a prover forge
openings).

Standard-domain bn254 dtypes, matching the KZG SRS fixture convention."""

from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from jax import Array, lax

from zorch.pcs.ipa.setup import IpaKey, setup

SF = zk_dtypes.bn254_sf
G1 = zk_dtypes.bn254_g1_affine


def _generator() -> Array:
    e = zk_dtypes.ecinfo(G1)
    return jnp.asarray(G1((e.gx, e.gy)))


def toy_key(n: int, seed: int = 1) -> IpaKey:
    """An `n`-element basis `G` and generator `U` from known scalars (`G_i =
    (seed + i + 1)·g1`, `U = (seed + n + 1)·g1`). Insecure by construction — the
    discrete logs are known — but enough to exercise the commit/open/verify
    round trip."""
    g1 = _generator()
    basis = [
        lax.convert_element_type(jnp.array(seed + i + 1, SF) * g1, G1)
        for i in range(n)
    ]
    u = lax.convert_element_type(jnp.array(seed + n + 1, SF) * g1, G1)
    return setup(jnp.stack(basis), u)
