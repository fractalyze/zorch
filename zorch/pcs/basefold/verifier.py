# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed root and checks
the fold consistency plus the jagged opening sumchecks. It holds only the
public params (`code` for the block geometry and fold, `tree` for the Merkle
config) — never the prover's retained codeword.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.config import (
    BasefoldCommitment,
    BasefoldProof,
    sample_rlc_coeffs,
)
from zorch.pcs.fold import sample_positions, verify_openings
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
    # Jitted verify body: an eager replay interprets each composite op-by-op
    # in Python (issue #140).
    _verify_body: Callable[..., tuple[Array, Transcript]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_verify_body", jax.jit(self._verify_traced))

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
        num_vars = z.shape[0]
        # Fail loud on a structurally malformed proof — a short message/layer list
        # would otherwise let the round loop silently skip checks (cf. the same
        # guard in `round.VerifyChain`). Eager guards, ahead of the jit zone.
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
        return self._verify_body(commitment, z, values, proof, transcript)

    def _verify_traced(
        self,
        commitment: BasefoldCommitment,
        z: Array,
        values: Array,
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        dtype = z.dtype
        K = values.shape[0]
        n = self.code.block_len
        num_vars = z.shape[0]
        t = transcript
        # Bind the commitment root into the transcript (mirrors `open`).
        t = t.observe(commitment)

        # Re-derive the RLC coeffs + batched claim (mirror open).
        t = t.observe(values)
        t, coeffs = sample_rlc_coeffs(t, K, dtype)
        current_claim = (values * coeffs).sum()

        # Replay the sumcheck reduction + fold challenges. The first num_vars-1
        # rounds are homogeneous — each observes the next fold layer's committed
        # root — so they ride one lax.scan; the final round is peeled because it
        # observes the cleartext final poly, not a root. The verifier's replay is
        # scan-shaped where the prover's fold_rounds is not: the prover *commits* a
        # shrinking layer each round (ragged), the verifier only *observes*
        # equal-shape roots (docs/conventions.md "Loops").
        one = jnp.ones((), dtype)
        z_rev = z[::-1]  # z_rev[r] binds the variable folded in round r
        zero_vals = jnp.stack([m[0] for m in proof.univariate_messages])
        one_vals = jnp.stack([m[1] for m in proof.univariate_messages])

        def fold_round(
            carry: tuple[Transcript, Array, Array],
            xs: tuple[Array, Array, Array, Array],
        ) -> tuple[tuple[Transcript, Array, Array], Array]:
            t, claim, ok = carry
            zero_val, one_val, last, root = xs
            expected = (one - last) * zero_val + last * one_val
            ok = ok & (claim == expected)
            t = t.observe(jnp.stack([zero_val, one_val]))
            t, beta = t.sample()
            beta = beta.reshape(())
            t = t.observe(root)
            return (t, zero_val + beta * one_val, ok), beta

        ok = jnp.bool_(True)
        betas: list[Array] = []
        if num_vars > 1:
            (t, current_claim, ok), head_betas = lax.scan(
                fold_round,
                (t, current_claim, ok),
                (zero_vals[:-1], one_vals[:-1], z_rev[:-1], jnp.stack(proof.fri_roots)),
            )
            # Index, don't iterate: list(field_array) dispatches lax.sign under
            # CUDA (cf. fri/prover._eval_poly).
            betas = [head_betas[r] for r in range(num_vars - 1)]

        # Peel the final round: it observes the cleartext final poly and runs the
        # IOPP terminal membership check instead of committing another layer.
        zero_val, one_val = zero_vals[-1], one_vals[-1]
        expected = (one - z_rev[-1]) * zero_val + z_rev[-1] * one_val
        ok = ok & (current_claim == expected)
        t = t.observe(jnp.stack([zero_val, one_val]))
        t, final_beta = t.sample()
        final_beta = final_beta.reshape(())
        t = t.observe(proof.final_poly)
        current_claim = zero_val + final_beta * one_val
        betas.append(final_beta)
        ok = ok & self.code.check_final(proof.final_poly, current_claim)

        # Query phase (pair layout from the code seam, mirrors
        # fri/verifier._verify_one).
        t, positions = sample_positions(t, n, self.num_queries)
        a = self.code.layer_positions(positions, num_vars)

        # Every opened pair must rebuild its layer's committed root: layer 0's
        # against the commitment, each fold layer's against its fri root. Collect
        # the lo/hi legs and reconstruct them in one batched pass (#163).
        lo0, hi0 = self.code.pair_indices(a[0], 0)
        legs = [
            (commitment, lo0, proof.component_opening.lo),
            (commitment, hi0, proof.component_opening.hi),
        ]
        for layer in range(1, num_vars):
            lo_idx, hi_idx = self.code.pair_indices(a[layer], layer)
            root = proof.fri_roots[layer - 1]
            legs.append((root, lo_idx, proof.query_openings[layer - 1].lo))
            legs.append((root, hi_idx, proof.query_openings[layer - 1].hi))
        ok = ok & verify_openings(self.tree, legs)

        # Rebuild layer-0 batched values from the opened ORIGINAL rows (RLC),
        # then fold each layer and check it reaches the next layer / final poly.
        # This loop stays unrolled (unlike the FS replay above): fold_values is
        # one lax.fft plus a few field ops per level, so it already traces in
        # O(1) per round — scanning it would add a control-flow boundary for no
        # trace+lower win (docs/conventions.md "Loops").
        lo_val = (proof.component_opening.lo.row * coeffs).sum(axis=-1)  # (Q,)
        hi_val = (proof.component_opening.hi.row * coeffs).sum(axis=-1)
        for i in range(num_vars):
            if i > 0:
                lo_val = proof.query_openings[i - 1].lo.row[:, 0]
                hi_val = proof.query_openings[i - 1].hi.row[:, 0]
            folded = self.code.fold_values(lo_val, hi_val, betas[i], a[i], i)
            if i < num_vars - 1:
                # The fold lands at a[i] in layer i+1 — the lo or hi of
                # that layer's opened pair, decided by the code's layout.
                next_lo_idx, _ = self.code.pair_indices(a[i + 1], i + 1)
                nxt = proof.query_openings[i]
                expected = jnp.where(
                    a[i] == next_lo_idx, nxt.lo.row[:, 0], nxt.hi.row[:, 0]
                )
            else:
                expected = proof.final_poly[a[i]]
            ok = ok & jnp.all(folded == expected)
        return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[BasefoldCommitment, BasefoldProof]] = BasefoldVerifier
