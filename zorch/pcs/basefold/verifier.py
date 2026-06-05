# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed root and checks
the fold consistency plus the jagged opening sumchecks. It holds only the
public params (`code` for the block geometry and fold, `tree` for the Merkle
config) — never the prover's retained codeword.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.basefold.config import (
    BasefoldCommitment,
    BasefoldProof,
    sample_rlc_coeffs,
)
from zorch.pcs.fold import query_layer_indices, sample_positions
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class BasefoldVerifier:
    """BaseFold PCS verifier (`PcsVerifier`)."""

    code: FoldableCode
    tree: MerkleTree
    # Must match the prover's; placeholder count, not soundness-calibrated.
    num_queries: int = 4

    def verify(
        self,
        commitment: BasefoldCommitment,
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
        n = self.code.block_len
        num_vars = z.shape[0]
        # Fail loud on a structurally malformed proof — a short message/layer list
        # would otherwise let the round loop silently skip checks (cf. the same
        # guard in `round.VerifyChain`).
        if self.code.message_len != (1 << num_vars):
            raise ValueError(
                f"point dimension {num_vars} doesn't match message_len "
                f"{self.code.message_len} (expected 2^{num_vars})"
            )
        if (
            len(proof.univariate_messages) != num_vars
            or len(proof.fri_roots) != num_vars - 1
            or len(proof.query_openings) != num_vars - 1
        ):
            raise ValueError(
                f"malformed proof: expected {num_vars} sumcheck messages and "
                f"{num_vars - 1} fold layers, got {len(proof.univariate_messages)} / "
                f"{len(proof.fri_roots)} / {len(proof.query_openings)}"
            )
        t = transcript
        # Bind the commitment root into the transcript (mirrors `open`).
        t = t.observe(commitment)

        # Re-derive the RLC coeffs + batched claim (mirror open).
        t = t.observe(values)
        t, coeffs = sample_rlc_coeffs(t, K, dtype)
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
        ok = ok & self.code.check_final(proof.final_poly, current_claim)

        # Query phase (natural order, mirrors fri/verifier._verify_one).
        t, positions = sample_positions(t, n, self.num_queries)
        a = query_layer_indices(positions, n, num_vars)
        half0 = n >> 1

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
            if i > 0:
                lo_val = proof.query_openings[i - 1].lo.row[:, 0]
                hi_val = proof.query_openings[i - 1].hi.row[:, 0]
            folded = self.code.fold_values(lo_val, hi_val, betas[i], a[i], i)
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


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[BasefoldCommitment, BasefoldProof]] = BasefoldVerifier
