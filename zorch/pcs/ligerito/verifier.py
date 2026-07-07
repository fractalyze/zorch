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
from zorch.pcs.fold import from_base_field, verify_openings
from zorch.pcs.ligerito.choreography import FsChoreography
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import MakeCode
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.sumcheck.prover import fold as sc_fold
from zorch.sumcheck.verifier import CompressedCoeffsSumcheckRound, SumcheckRound
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier

_ROUND = SumcheckRound(degree=2)
_COMPRESSED_ROUND = CompressedCoeffsSumcheckRound()


@dataclass(frozen=True)
class LigeritoVerifier:
    """Ligerito recursive PCS verifier. Mirrors `LigeritoProver`'s `make_code` /
    `tree` / `config` / `choreography` (share the choreography instance with
    the prover — it fixes the Fiat-Shamir wire for both sides)."""

    make_code: MakeCode
    tree: MerkleTree
    config: LigeritoConfig
    choreography: FsChoreography = FsChoreography()

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
        num_messages = self.choreography.num_messages(cfg)
        if len(proof.sumcheck_messages) != num_messages:
            raise ValueError(
                f"malformed proof: expected {num_messages} sumcheck messages, "
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
    chor = verifier.choreography
    dtype = z.dtype
    one = jnp.ones((), dtype)
    round_ = _COMPRESSED_ROUND if cfg.compressed_sumcheck_messages else _ROUND

    B = expand_eq_to_hypercube(z, one)
    claim = value
    ok = jnp.bool_(True)

    t = chor.bind_statement(transcript, commitment, z, value)

    roots = [commitment] + list(proof.recursive_roots)  # root of M_j = roots[j]
    residual = proof.final_residual
    msg_idx = 0
    num_vars = cfg.num_vars
    for j in range(cfg.num_levels):
        k_j = cfg.fold_ks[j]
        challenges = []
        for i in range(k_j):
            msg = proof.sumcheck_messages[msg_idx]
            msg_idx += 1
            t, r = chor.fold_challenge(t, msg, j, i)
            claim, ok_round = round_.check_reduce(claim, msg, r)
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
            t = chor.observe_root(t, roots[j + 1])
        else:
            t = chor.observe_residual(t, residual)

        code_j = verifier._code(j, 1 << num_vars)
        t, positions = chor.sample_queries(t, code_j.block_len, cfg.queries[j])
        opening = proof.component_openings[j]
        ok = ok & verify_openings(verifier.tree, [(roots[j], positions, opening)])
        opened = from_base_field(opening.row, dtype, kappa_j)  # (Q, kappa_j)
        v = (opened * eqc[None, :]).sum(axis=1)  # (Q,) proximity LHS
        points_s = code_j.eval_point(positions)  # (Q, num_vars)

        if is_final:
            # Direct: the in-clear residual must reproduce each proximity claim,
            # and the sumcheck's terminal claim closes against Σ_x residual·B.
            expected = jax.vmap(lambda p: eval_mle(residual, p))(points_s)  # (Q,)
            ok = ok & jnp.all(expected == v)
            ok = ok & (claim == (residual * B).sum())
            break

        # Induce the batched proximity claim into the running sumcheck (mirror
        # the prover): recompute the eval-point basis and enforced sum, glue with
        # a fresh separation challenge.
        t, alpha = sample_staggered_coeffs(
            t, cfg.queries[j], dtype, lsb_first=cfg.alpha_lsb_first
        )
        alpha = alpha[: cfg.queries[j]]
        eqps = jax.vmap(lambda p: expand_eq_to_hypercube(p, one))(points_s)
        b_new = (alpha[:, None] * eqps).sum(axis=0)  # (2^num_vars,)
        h_new = (alpha * v).sum()
        t, sep = t.sample()
        sep = sep.reshape(())
        B = B + sep * b_new
        claim = claim + sep * h_new

    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[LigeritoCommitment, LigeritoProof]] = LigeritoVerifier
