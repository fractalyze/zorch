# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed root and checks
the FRI fold consistency plus the jagged opening sumchecks. It holds only the
public params (`rs` for the domain/blowup, `tree` for the Merkle config) — never
the prover's retained codeword.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.fri import eval_domain, fri_fold_values
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.basefold.config import BasefoldProof
from zorch.pcs.fri.config import query_layer_indices, sample_positions
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class BasefoldVerifier:
    """BaseFold PCS verifier (`PcsVerifier`)."""

    rs: ReedSolomon
    tree: MerkleTree
    # Must match the prover's; placeholder count, not soundness-calibrated.
    num_queries: int = 4

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrix at one shared point, got {len(points)}"
            )
        z = points[0]
        dtype = z.dtype
        K = values.shape[0]
        n = self.rs.block_len
        num_vars = z.shape[0]
        t = transcript

        # Re-derive the RLC coeffs + batched claim (mirror open).
        t = t.observe(values)
        nbv = log2_ceil_usize(K)
        if nbv > 0:
            t, s = t.sample(nbv)
            coeffs = expand_eq_to_hypercube(s, jnp.ones((), dtype))[:K]
        else:
            coeffs = jnp.ones(1, dtype)
        current_claim = (values * coeffs).sum()

        # Replay the sumcheck reduction + fold challenges.
        one = jnp.ones((), dtype)
        ok = jnp.bool_(True)
        betas = []
        for r in range(num_vars):
            zero_val, one_val = proof.univariate_messages[r]
            last = z[-(r + 1)]
            expected = (one - last) * zero_val + last * one_val
            ok = ok & (current_claim == expected)
            t = t.observe(jnp.stack([zero_val, one_val]))
            t, beta = t.sample()
            beta = beta.reshape(())
            betas.append(beta)
            current_claim = zero_val + beta * one_val
            if r < num_vars - 1:
                t = t.observe(proof.fri_roots[r])
            else:
                t = t.observe(proof.final_poly)
        ok = ok & jnp.all(proof.final_poly == proof.final_poly[0])
        ok = ok & (current_claim == proof.final_poly[0])

        # Query phase (natural order, mirrors fri/verifier._verify_one).
        t, positions = sample_positions(t, n, self.num_queries)
        a = query_layer_indices(positions, n, num_vars)
        half0 = n >> 1
        domains = [eval_domain(self.rs.dtype, n >> i) for i in range(num_vars)]

        def roots_ok(root: Array, idx: Array, opening: Opening) -> Array:
            recon = jax.vmap(self.tree.reconstruct_root)(idx, opening)
            return jnp.all(recon == root)

        ok = ok & roots_ok(commitment, a[0], proof.component_opening.lo)
        ok = ok & roots_ok(commitment, a[0] + half0, proof.component_opening.hi)
        for layer in range(1, num_vars):
            half = n >> (layer + 1)
            ok = ok & roots_ok(
                proof.fri_roots[layer - 1], a[layer], proof.query_openings[layer - 1].lo
            )
            ok = ok & roots_ok(
                proof.fri_roots[layer - 1],
                a[layer] + half,
                proof.query_openings[layer - 1].hi,
            )

        # Rebuild layer-0 batched values from the opened ORIGINAL rows (RLC),
        # then fold each layer and check it reaches the next layer / final poly.
        lo_val = (proof.component_opening.lo.row * coeffs).sum(axis=-1)  # (Q,)
        hi_val = (proof.component_opening.hi.row * coeffs).sum(axis=-1)
        for i in range(num_vars):
            d = domains[i]
            if i > 0:
                lo_val = proof.query_openings[i - 1].lo.row[:, 0]
                hi_val = proof.query_openings[i - 1].hi.row[:, 0]
            folded = fri_fold_values(lo_val, hi_val, betas[i], d[a[i]])
            if i < num_vars - 1:
                half_next = n >> (i + 2)
                nxt = proof.query_openings[i]
                expected = jnp.where(
                    a[i] < half_next, nxt.lo.row[:, 0], nxt.hi.row[:, 0]
                )
            else:
                expected = proof.final_poly[a[i]]
            ok = ok & jnp.all(folded == expected)
        return ok, t
