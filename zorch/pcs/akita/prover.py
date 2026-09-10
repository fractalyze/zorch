# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita opening prover: the claimed values, the partials, and the fold — and
the virtual-to-committed reduction that feeds it.

Everything it sends comes from the digit witness `commit` retained — the field
partials by recomposing the digits, the response by folding them, the
reduction's factor tables the same way — so neither needs a second copy of the
polynomials. The layouts, transcript orders and the fold itself are the wire
and virtual modules', shared with the verifier.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from zorch.byte_transcript import ByteTranscript
from zorch.pcs.akita.challenge import BoundedChallengePolicy
from zorch.pcs.akita.commit import AkitaCommitter, AkitaProverData
from zorch.pcs.akita.config import AkitaConfig
from zorch.pcs.akita.virtual import (
    VirtualClaim,
    VirtualProof,
    factor_slices,
    observe_factors,
    observe_round,
    observe_virtual_claim,
    reduced_claim,
    shared_vars,
    squeeze_field,
)
from zorch.pcs.akita.wire import (
    AkitaOpeningClaim,
    AkitaOpeningProof,
    claimed_value,
    digits_to_field,
    draw_fold_challenges,
    fold_blocks,
    message_rows,
    observe_partials,
    observe_responses,
    observe_statement,
    packed_layout,
    require_exact_fold,
)
from zorch.pcs.stage import OpeningProof, OpeningWitness
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.domain import fold, natural_domain, summand_evals
from zorch.sumcheck.prover import ProductSummand


class AkitaProver:
    """Opens what an `AkitaCommitter` committed, under one challenge policy.

    The committer supplies the parameter point and the inner matrix the
    response is folded against; the policy is the consumer's soundness choice,
    and its verifier must hold the same one.
    """

    def __init__(
        self, committer: AkitaCommitter, policy: BoundedChallengePolicy
    ) -> None:
        require_exact_fold(committer.config, policy)
        self.committer = committer
        self.policy = policy

    def open(
        self,
        claim: AkitaOpeningClaim,
        witness: OpeningWitness[AkitaProverData],
        transcript: ByteTranscript,
    ) -> tuple[OpeningProof[AkitaOpeningProof], ByteTranscript]:
        config = self.committer.config
        ring = config.profile.ring
        degree = config.profile.degree
        layout = packed_layout(config, claim)
        data = witness.prover_data
        coefficients = digits_to_field(
            ring,
            data.witness,
            config.inner_decomposition.log_base,
            claim.points[0].dtype,
        )

        values: dict[int, Array] = {}
        partials = []
        for group in layout:
            inner, position, block = group.weights(degree)
            member_partials = []
            for index, message in zip(group.members, group.messages):
                table = group.view(config, message, coefficients)
                partial = (position[None, :, None] * table).sum(axis=1)
                values[index] = claimed_value(inner, block, partial)
                member_partials.append(partial)
            partials.append(fnp.stack(member_partials))
        claimed = fnp.stack([values[index] for index in range(len(claim.messages))])

        transcript = observe_statement(transcript, claim, claimed, data.images)
        transcript = observe_partials(transcript, partials)
        transcript, challenges = draw_fold_challenges(
            transcript, layout, self.policy, degree
        )
        blocks = ring.ntt(data.witness)
        responses = tuple(
            ring.intt(fold_blocks(ring, config, blocks, group, drawn))
            for group, drawn in zip(layout, challenges)
        )
        transcript = observe_responses(transcript, responses)
        proof = AkitaOpeningProof(data.images, tuple(partials), responses)
        return OpeningProof(claimed, proof), transcript

    def reduce_virtual(
        self,
        claim: VirtualClaim,
        witness: OpeningWitness[AkitaProverData],
        transcript: ByteTranscript,
    ) -> tuple[AkitaOpeningClaim, VirtualProof, ByteTranscript]:
        """Reduce a claim on a virtual product to one committed claim per
        factor, ready for `open` on the transcript this returns."""
        config = self.committer.config
        rounds = shared_vars(config, claim)
        dtype = claim.point.dtype
        coefficients = digits_to_field(
            config.profile.ring,
            witness.prover_data.witness,
            config.inner_decomposition.log_base,
            dtype,
        )
        one = fnp.ones((), dtype)
        slices = factor_slices(claim)
        # Each factor with its own slice bound at the claim's point: what is
        # left is a table over the shared variables alone, the one the
        # sumcheck folds.
        factors = []
        for message, (start, stop) in zip(claim.messages, slices):
            table = _message_table(config, coefficients, message)
            weights = expand_eq_to_hypercube(claim.point[start:stop], one)
            grid = table.reshape(1 << (stop - start), 1 << rounds)
            factors.append((weights[:, None] * grid).sum(axis=0))
        shared = expand_eq_to_hypercube(claim.point[slices[-1][1] :], one)
        state = fnp.stack([shared, *factors])
        summand = ProductSummand(len(factors) + 1)
        domain = natural_domain(summand.degree, dtype)

        transcript = observe_virtual_claim(transcript, claim)
        round_polys = []
        challenges = []
        for _ in range(rounds):
            round_poly = summand_evals(state, summand._combine, domain)
            transcript = observe_round(transcript, round_poly)
            transcript, challenge = squeeze_field(transcript, dtype)
            state = fold(state, challenge)
            round_polys.append(round_poly)
            challenges.append(challenge)
        values = state[1:, 0]
        transcript = observe_factors(transcript, values)
        sent = (
            fnp.stack(round_polys)
            if round_polys
            else fnp.zeros((0, summand.degree + 1), dtype)
        )
        return reduced_claim(claim, challenges), VirtualProof(sent, values), transcript


def _message_table(config: AkitaConfig, coefficients: Array, message: int) -> Array:
    """One message's field coefficients out of the per-block table, padding
    dropped."""
    flat = message_rows(config, message, coefficients).reshape(-1)
    return flat[: config.message_lens[message]]
