# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Curve descriptors and generator helper for the curve-PCS test fixtures.

A `Curve` pairs a G1's scalar field with its affine point dtype — the two things
a test needs to build a basis or SRS by `scalar · generator`. Shared so every
MSM-capable G1 PCS (IPA today, Pasta-cycle accumulation reuse next) draws its
fixtures from one place instead of re-declaring the dtypes. The curves named
here are production curves; what a *fixture* built from them may be is insecure
(a known-discrete-log basis), but that belongs to the fixture, not the curve."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import zk_dtypes
from jax import Array


class Curve(NamedTuple):
    """A G1 a curve PCS fixture can be built over: its scalar field and affine
    point dtype, both standard-domain (a basis is built by `scalar · generator`,
    and the test draws its scalars in Montgomery form, the production encoding)."""

    sf: type  # scalar field dtype
    g1: type  # G1 affine point dtype


BN254 = Curve(zk_dtypes.bn254_sf, zk_dtypes.bn254_g1_affine)
PALLAS = Curve(zk_dtypes.pallas_sf, zk_dtypes.pallas_g1_affine)


def generator(g1: type) -> Array:
    """The affine G1 generator of `g1`, from its `ecinfo`."""
    e = zk_dtypes.ecinfo(g1)
    return jnp.asarray(g1((e.gx, e.gy)))
