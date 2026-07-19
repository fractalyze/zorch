# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Toy IPA Pedersen basis fixture: scalar-mul a curve's G1 generator by known,
distinct seeds. Tests only — a real basis is hash-to-curve from public
randomness, with no known discrete logs (which here would let a prover forge
openings). The curve itself (bn254 today, the Pasta cycle for the accumulation
reuse) is production; only this known-DL basis is the toy.

The `Curve` descriptor lives in `zorch.pcs.testing.curves` so every G1 PCS shares
it. IPA is pairing-free, so — unlike the KZG SRS fixture, which stays
standard-domain only because `lax.pairing_check` decodes Montgomery inputs —
there is no encoding constraint here; the basis is built in standard domain and
the test draws its scalars in Montgomery form (the production encoding), the same
standard-basis / Montgomery-scalar split KZG's round-trip test uses."""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array, lax

from zorch.pcs.ipa.setup import IpaKey, setup
from zorch.pcs.testing.curves import Curve, generator


def toy_key(curve: Curve, n: int, seed: int = 1) -> IpaKey:
    """An `n`-element basis `G`, inner-product generator `U`, and blinding generator
    `S` from known scalars (`G_i = (seed + i + 1)·g1`, `U = (seed + n + 1)·g1`,
    `S = (seed + n + 2)·g1`) over `curve`. Insecure by construction — the discrete
    logs are known — but enough to exercise the commit/open/verify round trip,
    including the hiding/zk path (which blinds against `S`). The transparent path
    never reads `S`."""
    g1: Array = generator(curve.g1)
    basis = [
        lax.convert_element_type(fnp.array(seed + i + 1, curve.sf) * g1, curve.g1)
        for i in range(n)
    ]
    u = lax.convert_element_type(fnp.array(seed + n + 1, curve.sf) * g1, curve.g1)
    s = lax.convert_element_type(fnp.array(seed + n + 2, curve.sf) * g1, curve.g1)
    return setup(fnp.stack(basis), u, s)
