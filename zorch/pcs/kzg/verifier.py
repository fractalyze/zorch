# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KZG verifier: one multi-pairing check per opening.

The check `e(C − [f(z)]₁, [1]₂) = e(π, [τ − z]₂)` is rearranged so every scalar
combination lands on the G1 side (a `lax.msm`) and the G2 side is just the two
fixed verifier-key points — no G2 scalar-mul:

    e(C − f(z)·G1 + z·π, [1]₂) · e(−π, [τ]₂) == 1

This forces a **device split**: `lax.msm` is a GPU-only kernel while
`lax.pairing_check` legalizes to CPU only. The fork exposes no `device_put`, so the
G1 combinations are computed on the GPU, their coordinates pulled to the host, and
the points rebuilt on the CPU for the pairing — the split any pairing-based
verifier on this stack must make. The verifier is O(1), so the round-trip is
irrelevant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from jax import Array, lax

from zorch.pcs.kzg.setup import KzgVerifierKey
from zorch.transcript import Transcript


def _g1_to_host(point: Array, cpu: jax.Device) -> Array:
    x, y = np.array(point).item().raw
    with jax.default_device(cpu):
        return jnp.asarray(zk_dtypes.bn254_g1_affine((int(x), int(y))))


def _g2_to_host(point: Array, cpu: jax.Device) -> Array:
    (x0, x1), (y0, y1) = np.array(point).item().raw
    with jax.default_device(cpu):
        coords = ((int(x0), int(x1)), (int(y0), int(y1)))
        return jnp.asarray(zk_dtypes.bn254_g2_affine(coords))


@dataclass(frozen=True)
class KzgVerifier:
    vk: KzgVerifierKey

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: Array,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Check each opening's pairing equation; return `(all_ok, transcript)`."""
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == proof.shape[0] == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, proof={proof.shape[0]}"
            )
        one = jnp.array(1, dtype=zk_dtypes.bn254_sf)
        cpu = jax.devices("cpu")[0]
        gen_g2 = _g2_to_host(self.vk.gen_g2, cpu)
        tau_g2 = _g2_to_host(self.vk.tau_g2, cpu)
        oks = []
        for c, z, fz, pi in zip(commitment, points, values, proof):
            # G1 side (GPU msm): linear combos keep every scalar off G2.
            g1_combo = lax.msm(
                jnp.stack([one, -fz, z]), jnp.stack([c, self.vk.gen_g1, pi])
            )  # C − f(z)·G1 + z·π
            neg_pi = lax.msm(jnp.stack([-one]), jnp.stack([pi]))  # −π
            with jax.default_device(cpu):
                g1 = jnp.stack([_g1_to_host(g1_combo, cpu), _g1_to_host(neg_pi, cpu)])
                g2 = jnp.stack([gen_g2, tau_g2])
                oks.append(lax.pairing_check(g1, g2))
        return jnp.all(jnp.stack(oks)), transcript
