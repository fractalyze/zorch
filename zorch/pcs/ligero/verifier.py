# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Single-shot Ligero verifier — the `PcsVerifier` half of the matrix PCS.

`verify` replays the prover's Fiat-Shamir order (bind root, value, sent `w`, then
sample the query positions) and runs Ligero's two checks on the opened codeword
rows:

  * proximity  `<X[s], r_col> == encode(w)[s]`   for every sampled row `s`, and
  * value      `<r_row, w> == y`.

Proximity holds because `encode` is linear: `encode(w) = encode(X̃ · r_col) =
X · r_col`, so `encode(w)[s] = <X[s], r_col>` for an honest `w`; a forged `w`
disagrees with the committed codeword on ≥ `distance` positions and a random `s`
catches it. Value then reads `f(z) = r_row^T X̃ r_col = <r_row, w>`. The verifier
holds only the public params (`code` for the block geometry + the proximity
right-hand side `encode`, `tree` for the Merkle config) — never the prover's
matrix. Ligero needs only `LinearCode.encode`, not the fold seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as jnp
from frx import Array

from zorch.coding.linear_code import LinearCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.fold import from_base_field, sample_positions, verify_openings
from zorch.pcs.ligero.config import LigeroCommitment, LigeroProof
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class LigeroVerifier:
    """Single-shot Ligero PCS verifier (`PcsVerifier`)."""

    code: LinearCode
    tree: MerkleTree
    # Must match the prover's; placeholder count, not soundness-calibrated.
    num_queries: int = 4

    def verify(
        self,
        commitment: LigeroCommitment,
        points: Sequence[Array],
        value: Array,
        proof: LigeroProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Return `(ok, transcript)` where `ok` is a scalar boolean array."""
        if len(points) != 1:
            raise ValueError(f"Ligero opens at one point, got {len(points)}")
        z = points[0]
        num_vars = z.shape[0]
        k_row = log2_strict_usize(self.code.message_len)
        if num_vars < k_row:
            raise ValueError(
                f"point dimension {num_vars} is fewer than the row variables "
                f"{k_row} (= log2 message_len)"
            )
        if proof.w.shape[0] != self.code.message_len:
            raise ValueError(
                f"sent vector w has length {proof.w.shape[0]}, expected "
                f"rows={self.code.message_len} (= code.message_len)"
            )
        return _verify_body(self, commitment, z, value, proof, transcript)


# Jitted verify body: the verifier is the static key (by value, #214).
@partial(frx.jit, static_argnames=("verifier",))
def _verify_body(
    verifier: LigeroVerifier,
    commitment: Array,
    z: Array,
    value: Array,
    proof: LigeroProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    dtype = z.dtype
    one = jnp.ones((), dtype)
    n = verifier.code.block_len
    num_vars = z.shape[0]
    k_row = log2_strict_usize(verifier.code.message_len)
    cols = 1 << (num_vars - k_row)
    z_row, z_col = z[:k_row], z[k_row:]
    r_row = expand_eq_to_hypercube(z_row, one)  # (rows,)
    r_col = expand_eq_to_hypercube(z_col, one)  # (cols,)

    # Replay the prover's FS order: bind root, value, w, then sample positions.
    t = transcript.observe(commitment)
    t = t.observe(value)
    t = t.observe(proof.w)
    t, positions = sample_positions(t, n, verifier.num_queries)

    # Merkle: the opened rows rebuild the commitment at the query positions.
    merkle_ok = verify_openings(
        verifier.tree, [(commitment, positions, proof.component_opening)]
    )

    # Proximity: <X[s], r_col> == encode(w)[s]. encode(w) is one NTT over the
    # whole block; index it at the sampled positions. The committed leaves store
    # base-field limbs, so reinterpret each opened row to the value dtype first.
    opened = from_base_field(proof.component_opening.row, dtype, cols)  # (Q, cols)
    lhs = (opened * r_col[None, :]).sum(axis=1)  # (Q,)  broadcast-mul + sum, not `@`
    rhs = verifier.code.encode(proof.w)[positions]  # (Q,)
    proximity_ok = jnp.all(lhs == rhs)

    # Value: <r_row, w> == y.
    value_ok = (r_row * proof.w).sum() == value

    return merkle_ok & proximity_ok & value_ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[PcsVerifier[LigeroCommitment, LigeroProof]] = LigeroVerifier
