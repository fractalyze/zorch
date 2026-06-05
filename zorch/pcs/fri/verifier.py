# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI verifier: rebuild the quotient from `f`, then check fold consistency.

For each query the verifier reconstructs the quotient `g` at the conjugate pair
from the *committed* `f` values — `g(x) = (f(x) − v)/(x − z)` — so a false claim
`v ≠ f(z)` yields a non-low-degree `g` that fails both the per-layer fold check
and the final-layer constant check. It never trusts a prover-sent layer-0 oracle.
All arithmetic (NTT domain, field divide, Merkle rebuild) lowers on CPU and GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.commit.merkle import Opening
from zorch.pcs.fri.config import (
    FriCommitment,
    FriParams,
    FriProof,
    query_layer_indices,
    sample_positions,
)
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class FriVerifier:
    params: FriParams

    def verify(
        self,
        commitment: FriCommitment,
        points: Sequence[Array],
        values: Array,
        proof: Sequence[FriProof],
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == len(proof) == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, proof={len(proof)}"
            )
        t = transcript
        oks = []
        for f_root, z, v, pf in zip(commitment, points, values, proof):
            t, ok = self._verify_one(f_root, z, v, pf, t)
            oks.append(ok)
        return jnp.all(jnp.stack(oks)), t

    def _verify_one(
        self, f_root: Array, z: Array, v: Array, pf: FriProof, t: Transcript
    ) -> tuple[Transcript, Array]:
        params = self.params
        n = params.code.block_len

        # Replay Fiat-Shamir: fold challenges, then query positions.
        t = t.observe(f_root)
        betas = []
        for r in range(params.num_rounds):
            t, beta = t.sample()
            betas.append(beta.reshape(()))
            committed = r < params.num_rounds - 1
            t = t.observe(pf.layer_roots[r] if committed else pf.final_layer)
        t, positions = sample_positions(t, n, params.num_queries)
        a = query_layer_indices(positions, n, params.num_rounds)
        half0 = n >> 1

        # The final fold layer must be a constant (degree 0). FRI binds no
        # external claim, so the layer's own head is the claimed message.
        ok = params.code.check_final(pf.final_layer, pf.final_layer[0])

        # Merkle: every opened leaf must rebuild its committed root. vmap
        # reconstruct_root over the whole query batch and compare on device.
        def roots_ok(root: Array, idx: Array, opening: Opening) -> Array:
            recon = jax.vmap(params.tree.reconstruct_root)(idx, opening)
            return jnp.all(recon == root)

        ok = ok & roots_ok(f_root, a[0], pf.f_lo)
        ok = ok & roots_ok(f_root, a[0] + half0, pf.f_hi)
        for layer in range(1, params.num_rounds):
            half = n >> (layer + 1)
            root = pf.layer_roots[layer - 1]
            ok = ok & roots_ok(root, a[layer], pf.layers[layer - 1].lo)
            ok = ok & roots_ok(root, a[layer] + half, pf.layers[layer - 1].hi)

        # Rebuild the layer-0 quotient at each conjugate pair from f's leaves.
        d0 = params.code.domain()
        g_lo = (pf.f_lo.row[:, 0] - v) / (d0[a[0]] - z)
        g_hi = (pf.f_hi.row[:, 0] - v) / (d0[a[0] + half0] - z)

        for i in range(params.num_rounds):
            if i == 0:
                lo_val, hi_val = g_lo, g_hi
            else:
                lo_val = pf.layers[i - 1].lo.row[:, 0]
                hi_val = pf.layers[i - 1].hi.row[:, 0]
            folded = params.code.fold_values(lo_val, hi_val, betas[i], a[i], i)
            # The fold output lands at position a[i] in layer i+1 — the lo or hi of
            # that layer's conjugate pair depending on which half a[i] is in.
            if i < params.num_rounds - 1:
                half_next = n >> (i + 2)
                nxt = pf.layers[i]
                expected = jnp.where(
                    a[i] < half_next, nxt.lo.row[:, 0], nxt.hi.row[:, 0]
                )
            else:
                expected = pf.final_layer[a[i]]
            ok = ok & jnp.all(folded == expected)
        return t, ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/pcs.md "Instance anatomy".
    _: type[PcsVerifier[FriCommitment, list[FriProof]]] = FriVerifier
