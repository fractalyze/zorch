# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed roots and checks
the fold consistency of the batch open: the staggered RLC of the committed
matrices' opened rows must agree with the batched codeword's first pair-leaf, and
each fold layer's opened pair must fold to the next layer's, down to the constant
final poly. It holds only the public params (`code` for the block geometry and
fold, `tree` for the Merkle config) — never the prover's retained codeword.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.config import BasefoldCommitment, BasefoldProof
from zorch.pcs.fold import sample_positions, verify_fold_chain, verify_openings
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
        """Verify a single-matrix open — the degenerate one-round batch, the
        `PcsVerifier` seam shape."""
        return self.verify_batch([commitment], points, [values], proof, transcript)

    def verify_batch(
        self,
        commitments: Sequence[BasefoldCommitment],
        points: Sequence[Array],
        values: Sequence[Array],
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrices at one shared point, got {len(points)}"
            )
        if not (len(commitments) == len(values) == len(proof.component_openings)):
            raise ValueError(
                f"batch mismatch: {len(commitments)} commitments, {len(values)} "
                f"value vectors, {len(proof.component_openings)} component openings"
            )
        z = points[0]
        num_vars = z.shape[0]
        # Eager shape guards, ahead of the jit zone (mirrors the prover): with
        # zero variables the fold replay below would index an empty layer list.
        if num_vars < 1:
            raise ValueError("BaseFold opens over at least one variable, got none")
        if self.code.message_len != (1 << num_vars):
            raise ValueError(
                f"point dimension {num_vars} doesn't match message_len "
                f"{self.code.message_len} (expected 2^{num_vars})"
            )
        # Fail loud on a structurally malformed proof — a short message/layer list
        # would otherwise let the round loop silently skip checks.
        if (
            len(proof.univariate_messages) != num_vars
            or len(proof.fri_roots) != num_vars
            or len(proof.query_openings) != num_vars
        ):
            raise ValueError(
                f"malformed proof: expected {num_vars} sumcheck messages / fold "
                f"layers, got {len(proof.univariate_messages)} / "
                f"{len(proof.fri_roots)} / {len(proof.query_openings)}"
            )
        return _verify_batch_body(
            self, list(commitments), z, list(values), proof, transcript
        )


# Jitted verify body: an eager replay interprets each composite op-by-op in
# Python (issue #140). Module-level with the verifier as the static key — by
# value (#214), so same-config instances (one per test, in practice) share one
# trace.
@partial(jax.jit, static_argnames=("verifier",))
def _verify_batch_body(
    verifier: BasefoldVerifier,
    commitments: list[Array],
    z: Array,
    values: list[Array],
    proof: BasefoldProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    dtype = z.dtype
    n = verifier.code.block_len
    num_vars = z.shape[0]
    one = jnp.ones((), dtype)
    t = transcript

    # Re-derive the batch weights + initial claim (mirror open's FS order):
    # bind every commitment root, observe every matrix's claims, sample the
    # staggered coeffs, then bind the fold-round count.
    for root in commitments:
        t = t.observe(root)
    for vals in values:
        t = t.observe(vals)
    total_width = sum(int(v.shape[0]) for v in values)
    t, coeffs = sample_staggered_coeffs(t, total_width, dtype)
    current_claim = batch_staggered(list(values), coeffs)
    t = t.observe(jnp.asarray(num_vars, dtype))

    # Replay the interleaved sumcheck + fold challenges. Every round observes
    # its pre-fold pair-leaf commitment root before sampling β, so all
    # num_vars rounds are homogeneous (no peeled final round) and ride one
    # lax.scan — the poseidon2 permute markers stop scaling with num_vars
    # (#185). z_rev[r] binds the variable folded in round r.
    z_rev = z[::-1]
    zero_vals = jnp.stack([m[0] for m in proof.univariate_messages])
    one_vals = jnp.stack([m[1] for m in proof.univariate_messages])
    fri_roots = jnp.stack(proof.fri_roots)

    def fold_round(
        carry: tuple[Transcript, Array, Array],
        xs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Transcript, Array, Array], Array]:
        t, claim, ok = carry
        zero_val, one_val, last, root = xs
        expected = (one - last) * zero_val + last * one_val
        ok = ok & (claim == expected)
        t = t.observe(jnp.stack([zero_val, one_val]))
        t = t.observe(root)
        t, beta = t.sample()
        beta = beta.reshape(())
        return (t, zero_val + beta * one_val, ok), beta

    (t, current_claim, ok), betas_stacked = lax.scan(
        fold_round,
        (t, current_claim, jnp.bool_(True)),
        (zero_vals, one_vals, z_rev, fri_roots),
    )
    # Index, don't iterate: list(field_array) dispatches lax.sign under CUDA.
    betas = [betas_stacked[r] for r in range(num_vars)]

    # IOPP terminal membership: the fully folded codeword is the base-code
    # encoding of the final claim (a constant on the order-blowup domain).
    ok = ok & verifier.code.check_final(proof.final_poly, current_claim)

    # Bind the cleartext final codeword before sampling queries (mirror open).
    t = t.observe(proof.final_poly)

    # Query phase: shared positions; per-layer leaf index off the code seam.
    t, positions = sample_positions(t, n, verifier.num_queries)
    a = verifier.code.layer_positions(positions, num_vars)

    # Merkle: every component matrix rebuilds its commitment at the query
    # positions, every fold layer's pair-leaf rebuilds its fri root at the
    # halved index. One batched pass per leaf-row width (#163).
    legs = [
        (commitments[r], positions, proof.component_openings[r])
        for r in range(len(commitments))
    ]
    for i in range(num_vars):
        legs.append((proof.fri_roots[i], a[i], proof.query_openings[i]))
    ok = ok & verify_openings(verifier.tree, legs)

    # Fold consistency. The staggered RLC of the opened component rows is the
    # batched codeword at `positions`; it must be the right leg of fold layer
    # 0's opened pair. Then each layer's pair folds to the next layer's
    # opened value, down to the final poly.
    comp_val = batch_staggered(
        [co.row for co in proof.component_openings], coeffs
    )  # (Q,) batched value at `positions`
    lo0, _ = verifier.code.pair_indices(a[0], 0)
    leaf0 = proof.query_openings[0].row  # (Q, 2)
    in_leaf0 = jnp.where(positions == lo0, leaf0[:, 0], leaf0[:, 1])
    ok = ok & jnp.all(comp_val == in_leaf0)

    # Each layer's opened pair folds to the next layer's / the final poly.
    ok = ok & verify_fold_chain(
        verifier.code, proof.query_openings, betas, a, proof.final_poly
    )
    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[BasefoldCommitment, BasefoldProof]] = BasefoldVerifier
