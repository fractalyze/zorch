# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI verifier: rebuild the quotient from `f`, then check fold consistency.

For each query the verifier reconstructs the quotient `g` at the conjugate pair
from the *committed* `f` values — `g(x) = (f(x) − v)/(x − z)` — so a false claim
`v ≠ f(z)` yields a non-low-degree `g` that fails both the per-layer fold check
and the final-layer constant check. It never trusts a prover-sent layer-0 oracle.
All arithmetic (NTT domain, field divide, Merkle rebuild) lowers on CPU and GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.fold import sample_positions, verify_openings
from zorch.pcs.fri.config import FriCommitment, FriParams, FriProof
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class FriVerifier:
    params: FriParams
    # Jitted per-poly verify body (issue #140); one compile serves the batch.
    _verify_one_jit: Callable[..., tuple[Transcript, Array]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_verify_one_jit", jax.jit(self._verify_one))

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
        # Fail loud on a structurally short proof: the replay scan iterates over
        # whatever layer_roots it is handed, so a missing layer would silently
        # skip a round's checks rather than error (cf. the same guard in the
        # basefold verifier). Eager, ahead of the jit zone.
        rounds = self.params.num_rounds
        for pf in proof:
            if len(pf.layer_roots) != rounds - 1 or len(pf.layers) != rounds - 1:
                raise ValueError(
                    f"malformed proof: expected {rounds - 1} fold layers, got "
                    f"{len(pf.layer_roots)} roots / {len(pf.layers)} openings"
                )
        t = transcript
        oks = []
        for f_root, z, v, pf in zip(commitment, points, values, proof):
            t, ok = self._verify_one_jit(f_root, z, v, pf, t)
            oks.append(ok)
        return jnp.all(jnp.stack(oks)), t

    def _verify_one(
        self, f_root: Array, z: Array, v: Array, pf: FriProof, t: Transcript
    ) -> tuple[Transcript, Array]:
        params = self.params
        n = params.code.block_len

        # Replay Fiat-Shamir: fold challenges, then query positions. The first
        # num_rounds-1 rounds each observe the next layer's committed root, so
        # they ride one lax.scan; the final round observes the cleartext final
        # layer and is peeled (docs/conventions.md "Loops").
        t = t.observe(f_root)

        def fold_round(t: Transcript, root: Array) -> tuple[Transcript, Array]:
            t, beta = t.sample()
            t = t.observe(root)
            return t, beta.reshape(())

        betas: list[Array] = []
        if params.num_rounds > 1:
            t, head_betas = lax.scan(fold_round, t, jnp.stack(pf.layer_roots))
            # Index, don't iterate: list(field_array) dispatches lax.sign under
            # CUDA (cf. fri/prover._eval_poly).
            betas = [head_betas[r] for r in range(params.num_rounds - 1)]
        t, final_beta = t.sample()
        t = t.observe(pf.final_layer)
        betas.append(final_beta.reshape(()))
        t, positions = sample_positions(t, n, params.num_queries)
        a = params.code.layer_positions(positions, params.num_rounds)

        # The final fold layer must be a constant (degree 0). FRI binds no
        # external claim, so the layer's own head is the claimed message.
        ok = params.code.check_final(pf.final_layer, pf.final_layer[0])

        # Merkle: every opened pair must rebuild its committed root — layer 0's
        # against f's root, each fold layer's against its committed root.
        # Reconstruct all legs in one batched pass (#163).
        lo0, hi0 = params.code.pair_indices(a[0], 0)
        legs = [(f_root, lo0, pf.f_lo), (f_root, hi0, pf.f_hi)]
        for layer in range(1, params.num_rounds):
            lo_idx, hi_idx = params.code.pair_indices(a[layer], layer)
            root = pf.layer_roots[layer - 1]
            legs.append((root, lo_idx, pf.layers[layer - 1].lo))
            legs.append((root, hi_idx, pf.layers[layer - 1].hi))
        ok = ok & verify_openings(params.tree, legs)

        # Rebuild the layer-0 quotient at each point pair from f's leaves.
        d0 = params.code.domain()
        g_lo = (pf.f_lo.row[:, 0] - v) / (d0[lo0] - z)
        g_hi = (pf.f_hi.row[:, 0] - v) / (d0[hi0] - z)

        for i in range(params.num_rounds):
            if i == 0:
                lo_val, hi_val = g_lo, g_hi
            else:
                lo_val = pf.layers[i - 1].lo.row[:, 0]
                hi_val = pf.layers[i - 1].hi.row[:, 0]
            folded = params.code.fold_values(lo_val, hi_val, betas[i], a[i], i)
            # The fold output lands at position a[i] in layer i+1 — the lo or
            # hi of that layer's opened pair, decided by the code's layout.
            if i < params.num_rounds - 1:
                next_lo_idx, _ = params.code.pair_indices(a[i + 1], i + 1)
                nxt = pf.layers[i]
                expected = jnp.where(
                    a[i] == next_lo_idx, nxt.lo.row[:, 0], nxt.hi.row[:, 0]
                )
            else:
                expected = pf.final_layer[a[i]]
            ok = ok & jnp.all(folded == expected)
        return t, ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[FriCommitment, list[FriProof]]] = FriVerifier
