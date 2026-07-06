# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito recursive-open verifier — the `PcsVerifier` half of the recursive
matrix PCS. Replays the prover's continuous sumcheck and per-level Fiat-Shamir:
it reconstructs the running basis `B` (from the value
basis `eq(z)` plus each level's induced proximity basis, folded by the same
challenges), checks every sumcheck round's identity, rebuilds the queried
codeword rows from the committed roots, and recomputes each level's proximity
left-hand side `<M_j[s], eq(c_j)>` from the opened rows — inducing it into the
sumcheck exactly as the prover did. The final level's proximity is checked
directly against the in-clear residual, and the sumcheck's terminal claim must
equal `Σ_x residual(x)·B(x)`. It holds only the public params (the per-level codes
for block geometry + proximity points, `tree` for the Merkle config).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.tensor_code import TensorCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import sample_staggered_coeffs
from zorch.pcs.fold import from_base_field, sample_positions, verify_openings
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import MakeCode
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.sumcheck.prover import fold as sc_fold
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript
from zorch.utils.field import field_sum

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier

_ROUND = SumcheckRound(degree=2)


@dataclass(frozen=True)
class LigeritoVerifier:
    """Ligerito recursive PCS verifier. Mirrors `LigeritoProver`'s `make_code` /
    `tree` / `config`."""

    make_code: MakeCode
    tree: MerkleTree
    config: LigeritoConfig

    def _code(self, level: int, message_len: int) -> TensorCode:
        return self.make_code(message_len, self.config.log_inv_rates[level])

    def verify(
        self,
        commitment: LigeritoCommitment,
        points: Sequence[Array],
        value: Array,
        proof: LigeritoProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Return `(ok, transcript)` where `ok` is a scalar boolean array."""
        if len(points) != 1:
            raise ValueError(f"Ligerito opens at one point, got {len(points)}")
        z = points[0]
        cfg = self.config
        if z.shape[0] != cfg.num_vars:
            raise ValueError(
                f"point dimension {z.shape[0]} must equal the variable count "
                f"{cfg.num_vars}"
            )
        # Fail loud on a structurally malformed proof — a short list would let the
        # replay silently skip checks.
        if len(proof.sumcheck_messages) != sum(cfg.fold_ks):
            raise ValueError(
                f"malformed proof: expected {sum(cfg.fold_ks)} sumcheck messages, "
                f"got {len(proof.sumcheck_messages)}"
            )
        if len(proof.recursive_roots) != cfg.num_levels - 1:
            raise ValueError(
                f"malformed proof: expected {cfg.num_levels - 1} recursive roots, "
                f"got {len(proof.recursive_roots)}"
            )
        if len(proof.component_openings) != cfg.num_levels:
            raise ValueError(
                f"malformed proof: expected {cfg.num_levels} component openings, "
                f"got {len(proof.component_openings)}"
            )
        return _verify(self, commitment, z, value, proof, transcript)


def _verify(
    verifier: LigeritoVerifier,
    commitment: Array,
    z: Array,
    value: Array,
    proof: LigeritoProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    cfg = verifier.config
    dtype = z.dtype
    one = jnp.ones((), dtype)

    B = expand_eq_to_hypercube(z, one)
    claim = value
    ok = jnp.bool_(True)

    t = transcript.observe(commitment)
    t = t.observe(z)
    t = t.observe(value)

    roots = [commitment] + list(proof.recursive_roots)  # root of M_j = roots[j]
    residual = proof.final_residual
    msg_idx = 0
    num_vars = cfg.num_vars
    for j in range(cfg.num_levels):
        k_j = cfg.fold_ks[j]
        challenges = []
        for _ in range(k_j):
            msg = proof.sumcheck_messages[msg_idx]
            msg_idx += 1
            claim, t, r, ok_round = _ROUND(claim, msg, t)
            ok = ok & ok_round
            # The round verifier reduces only the claim; fold the public basis B
            # by the same challenge so it tracks the prover's folded B.
            B = sc_fold([B], r)[0]
            challenges.append(r)
        num_vars -= k_j
        eqc = expand_eq_to_hypercube(jnp.stack(challenges), one)  # (kappa_j,)
        kappa_j = 1 << k_j
        is_final = j == cfg.num_levels - 1

        if not is_final:
            t = t.observe(roots[j + 1])
        else:
            t = t.observe(residual)

        code_j = verifier._code(j, 1 << num_vars)
        t, positions = sample_positions(t, code_j.block_len, cfg.queries[j])
        opening = proof.component_openings[j]
        ok = ok & verify_openings(verifier.tree, [(roots[j], positions, opening)])
        opened = from_base_field(opening.row, dtype, kappa_j)  # (Q, kappa_j)
        v = field_sum(opened * eqc[None, :], axis=1)  # (Q,) proximity LHS
        points_s = code_j.eval_point(positions)  # (Q, num_vars)

        if is_final:
            # Direct: the in-clear residual must reproduce each proximity claim,
            # and the sumcheck's terminal claim closes against Σ_x residual·B.
            expected = jax.vmap(lambda p: eval_mle(residual, p))(points_s)  # (Q,)
            ok = ok & jnp.all(expected == v)
            ok = ok & (claim == field_sum(residual * B))
            break

        # Induce the batched proximity claim into the running sumcheck (mirror
        # the prover): recompute the eval-point basis and enforced sum, glue with
        # a fresh separation challenge.
        t, alpha = sample_staggered_coeffs(t, cfg.queries[j], dtype)
        alpha = alpha[: cfg.queries[j]]
        eqps = jax.vmap(lambda p: expand_eq_to_hypercube(p, one))(points_s)
        b_new = field_sum(alpha[:, None] * eqps, axis=0)  # (2^num_vars,)
        h_new = field_sum(alpha * v)
        t, sep = t.sample()
        sep = sep.reshape(())
        B = B + sep * b_new
        claim = claim + sep * h_new

    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[LigeritoCommitment, LigeritoProof]] = LigeritoVerifier
