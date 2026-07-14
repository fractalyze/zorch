# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KZG verifier: one multi-pairing check per opening.

The check `e(C − [f(z)]₁, [1]₂) = e(π, [τ − z]₂)` is rearranged so every scalar
combination lands on the G1 side (a `lax.msm`) and the G2 side is just the two
fixed verifier-key points — no G2 scalar-mul:

    e(C − f(z)·G1 + z·π, [1]₂) · e(−π, [τ]₂) == 1

This forces a **device split**: `lax.msm` is a GPU-only kernel while
`lax.pairing_check` legalizes to CPU only. frx exposes no `device_put`, so the
G1 combinations are computed on the GPU, their coordinates pulled to the host, and
the points rebuilt on the CPU for the pairing — the split any pairing-based
verifier on this stack must make. The verifier is O(1), so the round-trip is
irrelevant. The rebuild is domain-faithful (`raw` → `from_raw` of the same
dtype), so it is not the obstacle to Montgomery-form keys — `lax.pairing_check`'s
mont mis-decode is (fractalyze/zkx#518).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import frx
import frx.numpy as jnp
import numpy as np
from frx import Array, lax

from zorch.pcs.kzg.config import KzgCommitment, KzgProof
from zorch.pcs.kzg.setup import KzgVerifierKey
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


def _point_to_host(point: Array, cpu: frx.Device) -> Array:
    """Rebuild a device EC point on the CPU, preserving its dtype (and thus its
    domain): `raw` carries the dtype-domain limbs verbatim and `from_raw`
    reinterprets them under the same dtype."""
    pt = np.array(point).item()
    with frx.default_device(cpu):
        return jnp.asarray(type(pt).from_raw(pt.raw))


@dataclass(frozen=True)
class KzgVerifier:
    vk: KzgVerifierKey

    def verify(
        self,
        commitment: KzgCommitment,
        points: Sequence[Array],
        values: Array,
        proof: KzgProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Check each opening's pairing equation; return `(all_ok, transcript)`."""
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == proof.shape[0] == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, proof={proof.shape[0]}"
            )
        one = jnp.array(1, dtype=values.dtype)
        cpu = frx.devices("cpu")[0]
        gen_g2 = _point_to_host(self.vk.gen_g2, cpu)
        tau_g2 = _point_to_host(self.vk.tau_g2, cpu)
        oks = []
        for c, z, fz, pi in zip(commitment, points, values, proof):
            # G1 side (GPU msm): linear combos keep every scalar off G2.
            g1_combo = lax.msm(
                jnp.stack([one, -fz, z]), jnp.stack([c, self.vk.gen_g1, pi])
            )  # C − f(z)·G1 + z·π
            neg_pi = lax.msm(jnp.stack([-one]), jnp.stack([pi]))  # −π
            with frx.default_device(cpu):
                g1 = jnp.stack(
                    [_point_to_host(g1_combo, cpu), _point_to_host(neg_pi, cpu)]
                )
                g2 = jnp.stack([gen_g2, tau_g2])
                oks.append(lax.pairing_check(g1, g2))
        return jnp.all(jnp.stack(oks)), transcript


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[PcsVerifier[KzgCommitment, KzgProof]] = KzgVerifier
