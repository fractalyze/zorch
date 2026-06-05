# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI prover for univariate point evaluation (the DEEP quotient trick).

To open `f` at `z` with claim `v`, the prover shows the quotient
`g(x) = (f(x) − v)/(x − z)` is low degree: `g` is a polynomial of degree
`deg f − 1` exactly when `v = f(z)`. `g` is never committed separately — the
verifier rebuilds its codeword from the already-committed `f` at queried points —
so `open` Merkle-commits only the folded layers and threads the transcript through
the fold challenges. This is the structural opposite of KZG on the same PCS seam:
interactive, Merkle-backed, no SRS, and entirely field/NTT arithmetic that lowers
on both CPU and GPU (no MSM). The query phase is device-batched — positions are a
device int32 array and each Merkle opening is one `vmap` over them, no host indices
or per-query loop — while the per-round fold transcript stays host-driven pending a
device-side transcript (#3). Scope: a single base-field polynomial per opening,
fixed small parameters — a demonstration of the seam, not a hardened prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.reed_solomon import eval_domain, fri_fold
from zorch.commit.merkle import Opening
from zorch.pcs.fri.config import (
    FriParams,
    FriProof,
    LayerOpening,
    query_layer_indices,
    sample_positions,
)
from zorch.transcript import Transcript


def _eval_poly(coeffs: Array, z: Array) -> Array:
    """f(z) for ascending `coeffs`, by index-loop Horner (iterating a field array
    under CUDA dispatches lax.sign — index instead)."""
    acc = jnp.zeros((), coeffs.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * z + coeffs[i]
    return acc


@dataclass(frozen=True)
class FriProver:
    params: FriParams

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[Array, list[tuple[Array, Array, list[Array]]]]:
        """RS-encode each coefficient vector and Merkle-commit the codeword.
        Returns stacked roots and the prover data needed to open."""
        roots, data = [], []
        for coeffs in polys:
            codeword = self.params.code.encode(coeffs)
            matrix = codeword.reshape(-1, 1)
            root, digest_layers = self.params.tree.commit(matrix)
            roots.append(root)
            data.append((coeffs, matrix, digest_layers))
        return jnp.stack(roots), data

    def open(
        self,
        prover_data: Sequence[tuple[Array, Array, list[Array]]],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, list[FriProof], Transcript]:
        if len(prover_data) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data)} polys vs {len(points)} points"
            )
        values, proofs = [], []
        t = transcript
        for (coeffs, matrix, digest_layers), z in zip(prover_data, points):
            v = _eval_poly(coeffs, z)
            t, proof = self._open_one(coeffs, matrix, digest_layers, z, v, t)
            values.append(v)
            proofs.append(proof)
        return jnp.stack(values), proofs, t

    def _open_one(
        self,
        coeffs: Array,
        matrix: Array,
        digest_layers: list[Array],
        z: Array,
        v: Array,
        t: Transcript,
    ) -> tuple[Transcript, FriProof]:
        params = self.params
        n = params.code.block_len
        domain = eval_domain(params.code.dtype, n)
        f_codeword = matrix.reshape(-1)
        quotient = (f_codeword - v) / (domain - z)  # layer 0, rebuilt from f at verify

        t = t.observe(digest_layers[-1][0])  # bind the f commitment root
        betas, layer_mats, layer_dls, layer_roots = [], [], [], []
        cw = quotient
        final_layer = cw
        for r in range(params.num_rounds):
            t, beta = t.sample()
            beta = beta.reshape(())
            betas.append(beta)
            cw = fri_fold(cw, beta)
            if r < params.num_rounds - 1:
                m = cw.reshape(-1, 1)
                root, dl = params.tree.commit(m)
                layer_mats.append(m)
                layer_dls.append(dl)
                layer_roots.append(root)
                t = t.observe(root)
            else:
                final_layer = cw
                t = t.observe(final_layer)

        # Query phase, all on device: positions are an int32 array and every
        # Merkle opening is one vmap over that batch — no host indices, no
        # per-query Python loop.
        t, positions = sample_positions(t, n, params.num_queries)
        a = query_layer_indices(positions, n, params.num_rounds)
        half0 = n >> 1

        def open_batch(mtx: Array, dl: list[Array], idx: Array) -> Opening:
            return jax.vmap(lambda i: params.tree.open(mtx, dl, i))(idx)

        f_lo = open_batch(matrix, digest_layers, a[0])
        f_hi = open_batch(matrix, digest_layers, a[0] + half0)
        layer_opens = []
        for layer in range(1, params.num_rounds):
            half = n >> (layer + 1)
            m, dl = layer_mats[layer - 1], layer_dls[layer - 1]
            lo = open_batch(m, dl, a[layer])
            hi = open_batch(m, dl, a[layer] + half)
            layer_opens.append(LayerOpening(lo, hi))
        return t, FriProof(v, layer_roots, final_layer, f_lo, f_hi, layer_opens)
