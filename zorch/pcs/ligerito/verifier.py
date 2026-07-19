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

import frx.numpy as fnp
from frx import Array

from zorch.coding.tensor_code import TensorCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import sample_staggered_coeffs
from zorch.pcs.fold import from_base_field, verify_openings
from zorch.pcs.ligerito.basis import select_commit_basis
from zorch.pcs.ligerito.choreography import LigeritoChoreography
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import MakeCode
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck import prover as sc_prover
from zorch.sumcheck.domain import fold
from zorch.sumcheck.verifier import CompressedCoeffsSumcheckRound, SumcheckRound
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier

_ROUND = SumcheckRound(degree=2)
_COMPRESSED_ROUND = CompressedCoeffsSumcheckRound()
# Prover-round duals, for the eager policy's terminal pin: the last emitted
# message is the residual state's round poly, recomputable in the clear.
_P_ROUND = sc_prover.StandardRound(sc_prover.ProductSummand(degree=2))
_P_COMPRESSED_ROUND = sc_prover.CompressedProductRound()


@dataclass(frozen=True)
class LigeritoVerifier:
    """Ligerito recursive PCS verifier. Mirrors `LigeritoProver`'s `make_code` /
    `tree` / `config` / `choreography` (share the choreography instance with
    the prover — it fixes the Fiat-Shamir wire for both sides)."""

    make_code: MakeCode
    tree: MerkleTree
    config: LigeritoConfig
    choreography: LigeritoChoreography = LigeritoChoreography()

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
        if z.shape[0] != self.config.num_vars:
            raise ValueError(
                f"point dimension {z.shape[0]} must equal the variable count "
                f"{self.config.num_vars}"
            )
        self._check_shape(proof)
        one = fnp.ones((), z.dtype)
        return _verify(
            self,
            commitment,
            z,
            expand_eq_to_hypercube(z, one),
            value,
            proof,
            transcript,
        )

    def verify_with_basis(
        self,
        commitment: LigeritoCommitment,
        basis: Array,
        value: Array,
        proof: LigeritoProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """`verify` for a RAW hypercube basis instead of a point — the dual of
        `LigeritoProver.open_with_basis` (`bind_statement` receives
        `point=None`)."""
        if basis.shape[0] != 1 << self.config.num_vars:
            raise ValueError(
                f"basis length {basis.shape[0]} must be 2^{self.config.num_vars}"
            )
        self._check_shape(proof)
        return _verify(self, commitment, None, basis, value, proof, transcript)

    def _check_shape(self, proof: LigeritoProof) -> None:
        """Fail loud on a structurally malformed proof — a short list would let
        the replay silently skip checks."""
        cfg = self.config
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
        if len(proof.ood_values) != cfg.total_ood:
            raise ValueError(
                f"malformed proof: expected {cfg.total_ood} OOD values, "
                f"got {len(proof.ood_values)}"
            )
        num_pow = self.choreography.num_pow_witnesses(cfg)
        if len(proof.pow_witnesses) != num_pow:
            raise ValueError(
                f"malformed proof: expected {num_pow} proof-of-work witnesses, "
                f"got {len(proof.pow_witnesses)}"
            )


def _verify(
    verifier: LigeritoVerifier,
    commitment: Array,
    z: Array | None,
    B: Array,
    value: Array,
    proof: LigeritoProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    cfg = verifier.config
    chor = verifier.choreography
    dtype = B.dtype
    one = fnp.ones((), dtype)
    round_ = _COMPRESSED_ROUND if cfg.compressed_sumcheck_messages else _ROUND
    basis = select_commit_basis(cfg.monomial_commit)

    claim = value
    ok = fnp.bool_(True)

    t = chor.bind_statement(transcript, commitment, z, value)

    roots = [commitment] + list(proof.recursive_roots)  # root of M_j = roots[j]
    residual = proof.final_residual
    msgs = iter(proof.sumcheck_messages)
    oods = iter(proof.ood_values)
    wits = iter(proof.pow_witnesses)
    num_vars = cfg.num_vars

    def take(t: Transcript) -> tuple[Transcript, Array]:
        """The next eager emission off the wire, absorbed like the prover did."""
        m = next(msgs)
        return chor.observe_message(t, m), m

    def check_grind(t: Transcript, bits: int | None) -> Transcript:
        nonlocal ok
        if bits is None:
            return t
        t, ok_grind = chor.check_grind(t, bits, next(wits))
        ok = ok & ok_grind
        return t

    # Under the eager policy `cur` tracks the current round's message as the
    # prover emitted it: read off the proof after every fold, recombined
    # linearly at every glue (round polys are linear in the basis factor), so
    # each round checks against the same combined message the lazy wire would
    # have carried whole.
    eager = chor.eager_messages
    cur: Array | None = None
    if eager:
        t, cur = take(t)

    for j in range(cfg.num_levels):
        k_j = cfg.fold_ks[j]
        challenges = []
        for i in range(k_j):
            msg = cur if cur is not None else next(msgs)
            t = check_grind(t, chor.fold_grind_bits(j, i))
            t, r = chor.fold_challenge(t, None if eager else msg, j, i)
            claim, ok_round = round_.check_reduce(claim, msg, r)
            ok = ok & ok_round
            # The round verifier reduces only the claim; fold the public basis B
            # by the same challenge so it tracks the prover's folded B.
            B = fold(B, r)
            challenges.append(r)
            if eager:
                t, cur = take(t)
        num_vars -= k_j
        # The opened row's lane axis follows the commit basis: bit-reversed
        # under monomial_commit (lane bit j <-> challenge k_j-1-j), natural
        # otherwise. eq of the reversed challenge vector IS the bit-reversed
        # eq table, so one expansion serves both.
        lane_chals = challenges[::-1] if cfg.monomial_commit else challenges
        eqc = expand_eq_to_hypercube(fnp.stack(lane_chals), one)  # (kappa_j,)
        kappa_j = 1 << k_j
        is_final = j == cfg.num_levels - 1

        if not is_final:
            t = chor.observe_root(t, roots[j + 1])
            # OOD binding (mirror the prover): rebuild the drawn basis, take the
            # claimed value off the proof, and glue both into basis and claim —
            # the claim's honesty is enforced by the continuing sumcheck.
            for _ in range(cfg.ood_count(j)):
                t, zs = t.sample(num_vars)
                b_ood = expand_eq_to_hypercube(zs.astype(dtype), one)
                y = next(oods)
                t = t.observe(y)
                if cur is not None:
                    t, m = take(t)
                t, sep = t.sample()
                sep = sep.reshape(())
                B = B + sep * b_ood
                claim = claim + sep * y
                if cur is not None:
                    cur = cur + sep * m
        else:
            t = chor.observe_residual(t, residual)

        code_j = verifier._code(j, 1 << num_vars)
        t = check_grind(t, chor.query_grind_bits(j))
        t, positions = chor.sample_queries(t, code_j.block_len, cfg.queries[j])
        opening = proof.component_openings[j]
        ok = ok & verify_openings(verifier.tree, [(roots[j], positions, opening)])
        opened = from_base_field(opening.row, dtype, kappa_j)  # (Q, kappa_j)
        v = (opened * eqc[None, :]).sum(axis=1)  # (Q,) proximity LHS
        points_s = code_j.eval_point(positions)  # (Q, num_vars)

        if is_final:
            # Direct: the in-clear residual must reproduce each proximity claim,
            # and the sumcheck's terminal claim closes against Σ_x residual·B.
            bases = basis.proximity_basis(points_s, one)  # (Q, 2^nv)
            expected = (residual[None, :] * bases).sum(axis=1)  # (Q,)
            ok = ok & fnp.all(expected == v)
            ok = ok & (claim == (residual * B).sum())
            if cur is not None:
                # The eager wire's terminal emission is the residual state's
                # round poly — recompute it in the clear and pin it exactly.
                p_round = (
                    _P_COMPRESSED_ROUND
                    if cfg.compressed_sumcheck_messages
                    else _P_ROUND
                )
                ok = ok & fnp.all(cur == p_round._round_poly(fnp.stack([residual, B])))
            break

        # Induce the batched proximity claim into the running sumcheck (mirror
        # the prover): recompute the eval-point basis and enforced sum, glue with
        # a fresh separation challenge.
        t, alpha = sample_staggered_coeffs(
            t, cfg.queries[j], dtype, lsb_first=cfg.alpha_lsb_first
        )
        alpha = alpha[: cfg.queries[j]]
        eqps = basis.proximity_basis(points_s, one)
        b_new = (alpha[:, None] * eqps).sum(axis=0)  # (2^num_vars,)
        h_new = (alpha * v).sum()
        if cur is not None:
            t, m = take(t)
        t, sep = t.sample()
        sep = sep.reshape(())
        B = B + sep * b_new
        claim = claim + sep * h_new
        if cur is not None:
            cur = cur + sep * m

    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[PcsVerifier[LigeritoCommitment, LigeritoProof]] = LigeritoVerifier
