# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita opening verifier: the payload, the claimed values, and the three fold
relations, replayed from the proof alone — and the virtual-to-committed
reduction's sumcheck, replayed the same way.

The committer rides the verifier here, unlike the pairing instances: every
matrix it holds is one the verifier re-applies — the outer tier to check the
images against the payload, the inner tier to check the response against the
images — so a lattice PCS has no proving-only key to keep out of reach. It
never sees a witness.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import Array
from lattice_frx.ring import Coeff, Eval

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import commitments_equal, within_bound
from zorch.pcs.akita.challenge import BoundedChallengePolicy
from zorch.pcs.akita.commit import AkitaCommitter
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
    ClaimGroup,
    GroupChallenges,
    claimed_value,
    digits_to_field,
    draw_fold_challenges,
    fold_blocks,
    observe_partials,
    observe_responses,
    observe_statement,
    packed_layout,
    require_exact_fold,
)
from zorch.pcs.stage import OpeningProof
from zorch.poly.eq import eval_eq
from zorch.sumcheck.reduce import reduce_evals


class AkitaVerifier:
    """Checks an `AkitaProver` opening against the payload it claims."""

    def __init__(
        self, committer: AkitaCommitter, policy: BoundedChallengePolicy
    ) -> None:
        require_exact_fold(committer.config, policy)
        self.committer = committer
        self.policy = policy

    def verify(
        self,
        claim: AkitaOpeningClaim,
        proof: OpeningProof[AkitaOpeningProof],
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Accept iff the images re-commit to the payload and every group's
        values, norm, inner relation and evaluation relation hold.

        The transcript advances over the whole proof either way, so what a
        caller continues on depends on the proof bytes and not on the verdict.
        A proof of the wrong shape is a malformed wire rather than a false
        statement, and raises.
        """
        config = self.committer.config
        ring = config.profile.ring
        layout = packed_layout(config, claim)
        body = proof.proof
        self._require_shapes(claim, layout, proof)

        transcript = observe_statement(transcript, claim, proof.values, body.images)
        transcript = observe_partials(transcript, body.partials)
        transcript, challenges = draw_fold_challenges(
            transcript, layout, self.policy, config.profile.degree
        )
        transcript = observe_responses(transcript, body.responses)

        payload = self.committer.outer_commit(body.images)
        if not commitments_equal(payload, claim.commitment):
            return False, transcript
        images = ring.ntt(body.images)
        for group, partials, response, drawn in zip(
            layout, body.partials, body.responses, challenges
        ):
            if not self._group_holds(
                group, proof.values, partials, response, images, drawn
            ):
                return False, transcript
        return True, transcript

    def reduce_virtual(
        self,
        claim: VirtualClaim,
        proof: VirtualProof,
        transcript: ByteTranscript,
    ) -> tuple[AkitaOpeningClaim, Array, bool, ByteTranscript]:
        """Replay the reduction: the committed claim it leaves, the values that
        claim's opening must return, and whether the sumcheck held.

        The opening does not know it serves a reduction, so the link is the
        caller's to close: accept only if the opening of the returned claim
        verifies *and* its values equal the returned ones.
        """
        rounds = shared_vars(self.committer.config, claim)
        dtype = claim.point.dtype
        degree = len(claim.messages) + 1
        _require_field(
            "reduce_virtual: rounds", proof.rounds, (rounds, degree + 1), dtype
        )
        _require_field(
            "reduce_virtual: factors", proof.factors, (len(claim.messages),), dtype
        )

        transcript = observe_virtual_claim(transcript, claim)
        running = claim.value
        holds = True
        challenges = []
        for index in range(rounds):
            round_poly = proof.rounds[index]
            transcript = observe_round(transcript, round_poly)
            transcript, challenge = squeeze_field(transcript, dtype)
            running, round_holds = reduce_evals(running, round_poly, challenge, degree)
            holds = holds and bool(round_holds)
            challenges.append(challenge)
        transcript = observe_factors(transcript, proof.factors)

        final = fnp.prod(proof.factors)
        if challenges:
            shared = claim.point[factor_slices(claim)[-1][1] :]
            final = final * eval_eq(shared, fnp.stack(challenges))
        holds = holds and _field_equal(running, final)
        return reduced_claim(claim, challenges), proof.factors, holds, transcript

    def _group_holds(
        self,
        group: ClaimGroup,
        values: Array,
        partials: Array,
        response: Coeff,
        images: Eval,
        drawn: GroupChallenges,
    ) -> bool:
        committer = self.committer
        config = committer.config
        ring = config.profile.ring
        degree = config.profile.degree
        inner, position, block = group.weights(degree)
        for index, partial in zip(group.members, partials):
            if not _field_equal(values[index], claimed_value(inner, block, partial)):
                return False

        bound = config.inner_beta_inf * sum(
            int(np.abs(challenge).sum()) for member in drawn for challenge in member
        )
        if not within_bound("verify", ring, response, bound):
            return False

        recommitted = committer.inner.commit_batch(
            committer.inner_matrix, ring.ntt(response)
        )
        if not commitments_equal(
            recommitted, fold_blocks(ring, config, images, group, drawn)
        ):
            return False

        folded = digits_to_field(
            ring, response, config.inner_decomposition.log_base, group.point.dtype
        )
        contracted = (position[:, None] * folded).sum(axis=0)
        return _field_equal(contracted, _negacyclic_fold(drawn, partials))

    def _require_shapes(
        self,
        claim: AkitaOpeningClaim,
        layout: tuple[ClaimGroup, ...],
        proof: OpeningProof[AkitaOpeningProof],
    ) -> None:
        config = self.committer.config
        body = proof.proof
        dtype = claim.points[0].dtype
        _require_field("verify: values", proof.values, (len(claim.messages),), dtype)
        _require_coeff(
            "verify: images", body.images, (config.blocks, config.inner_rows)
        )
        if len(body.partials) != len(layout) or len(body.responses) != len(layout):
            raise ValueError(
                f"verify: {len(body.partials)} partials and {len(body.responses)} "
                f"responses for {len(layout)} groups"
            )
        for group, partial, response in zip(layout, body.partials, body.responses):
            want = (len(group.messages), group.super_blocks, config.profile.degree)
            _require_field("verify: partials", partial, want, dtype)
            _require_coeff(
                "verify: response", response, (group.positions, config.inner_cols)
            )


def _require_field(name: str, array: Array, shape: tuple[int, ...], dtype: Any) -> None:
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(
            f"{name} {tuple(array.shape)} {array.dtype}, want {shape} {dtype}"
        )


def _require_coeff(name: str, element: Coeff, lead: tuple[int, ...]) -> None:
    """A coefficient-domain element of the declared module shape — the norm
    and the lift that read it are coefficient-domain notions."""
    if not isinstance(element, Coeff):
        raise TypeError(f"{name} must be Coeff, got {type(element).__name__}")
    got = tuple(element.limbs[0].shape[:-1])
    if got != lead:
        raise ValueError(f"{name} leading axes {got}, want {lead}")


def _negacyclic_fold(challenges: GroupChallenges, partials: Array) -> Array:
    """`Σ_member Σ_s c_s·E_s` in `F[X]/(X^d+1)` as one product and one
    reduction.

    Each short integer challenge becomes its signed negacyclic matrix on the
    host — entry `(j, i)` is `c_{j-i}`, negated where `j - i` wraps below zero
    — and enters the field through Python integers: a negative integer scaled
    into a field array wraps as a u64 instead of negating.
    """
    degree = partials.shape[-1]
    rows = np.arange(degree)[:, None]
    cols = np.arange(degree)[None, :]
    wrap = (rows - cols) % degree
    sign = np.where(rows >= cols, 1, -1)
    matrices = np.array(
        [[challenge[wrap] * sign for challenge in drawn] for drawn in challenges],
        dtype=np.int64,
    )
    lifted = fnp.asarray(matrices.astype(object).astype(partials.dtype))
    return (lifted * partials[:, :, None, :]).sum(axis=(0, 1, 3))


def _field_equal(a: Array, b: Array) -> bool:
    return bool(
        np.array_equal(np.asarray(a).astype(object), np.asarray(b).astype(object))
    )
