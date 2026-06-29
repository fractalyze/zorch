# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Toy IPA Pedersen basis fixture: scalar-mul a curve's G1 generator by known,
distinct seeds. Tests only — a real basis is hash-to-curve from public
randomness, with no known discrete logs (which here would let a prover forge
openings).

Curve-parameterized via `ToyCurve` so the same fixture serves every MSM-capable
G1 the seam runs on (bn254 today, the Pasta cycle for the accumulation reuse).
IPA is pairing-free, so — unlike the KZG SRS fixture, which stays standard-domain
only because `lax.pairing_check` decodes Montgomery inputs — there is no encoding
constraint here; the basis is built in standard domain and the test draws its
scalars in Montgomery form (the production encoding), the same standard-basis /
Montgomery-scalar split KZG's round-trip test uses."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import zk_dtypes
from jax import Array, lax

from zorch.pcs.ipa.setup import IpaKey, setup


class ToyCurve(NamedTuple):
    """A G1 the toy basis can be built over: its scalar field and affine point
    dtype, both standard-domain (the basis is built by `scalar · generator`)."""

    sf: type  # scalar field dtype
    g1: type  # G1 affine point dtype


BN254 = ToyCurve(zk_dtypes.bn254_sf, zk_dtypes.bn254_g1_affine)
PALLAS = ToyCurve(zk_dtypes.pallas_sf, zk_dtypes.pallas_g1_affine)


def _generator(g1: type) -> Array:
    e = zk_dtypes.ecinfo(g1)
    return jnp.asarray(g1((e.gx, e.gy)))


def toy_key(curve: ToyCurve, n: int, seed: int = 1) -> IpaKey:
    """An `n`-element basis `G` and generator `U` from known scalars (`G_i =
    (seed + i + 1)·g1`, `U = (seed + n + 1)·g1`) over `curve`. Insecure by
    construction — the discrete logs are known — but enough to exercise the
    commit/open/verify round trip."""
    g1 = _generator(curve.g1)
    basis = [
        lax.convert_element_type(jnp.array(seed + i + 1, curve.sf) * g1, curve.g1)
        for i in range(n)
    ]
    u = lax.convert_element_type(jnp.array(seed + n + 1, curve.sf) * g1, curve.g1)
    return setup(jnp.stack(basis), u)
