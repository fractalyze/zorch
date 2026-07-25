# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Separately deployable Spartan prover and verifier roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.challenge import DEFAULT_CHALLENGES, ChallengePolicy
from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.spartan.lincheck import (
    ColumnEvaluationClaim,
    InnerProof,
    InnerProver,
    InnerVerifier,
    LincheckClaim,
    LincheckWitness,
    batch_claims,
)
from zorch.spartan.pcs_glue import (
    WitnessOpeningClaim,
    WitnessOpeningWitness,
    WitnessOpenProof,
    WitnessOpenProver,
    WitnessOpenVerifier,
    witness_opening_claim,
)
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import (
    OuterProof,
    OuterProver,
    OuterVerifier,
    RowEvaluationClaim,
    ZerocheckClaim,
    ZerocheckWitness,
)
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.transcript import Transcript


@dataclass(frozen=True)
class SpartanClaim:
    """Public R1CS satisfiability claim."""

    instance: R1CS
    public_inputs: Array

    def __post_init__(self) -> None:
        if self.public_inputs.shape != (self.instance.num_io,):
            raise ValueError(
                f"expected {self.instance.num_io} public inputs, "
                f"got shape {self.public_inputs.shape}"
            )


@dataclass(frozen=True)
class SpartanWitness:
    """Private assignment witnessing a ``SpartanClaim``."""

    assignment: Array


@dataclass(frozen=True)
class SpartanProof:
    """One named reduction-proof section per coarse protocol phase."""

    commitment: Array
    outer: OuterProof
    inner: InnerProof
    witness_open: WitnessOpenProof


def _observe_framed(transcript: Transcript, tag: int, values: Array) -> Transcript:
    header = fnp.array([tag, values.ndim, *values.shape], values.dtype)
    transcript = transcript.observe(header)
    if values.size > 0:
        transcript = transcript.observe(fnp.reshape(values, (-1,)))
    return transcript


def _absorb_claim(
    transcript: Transcript,
    claim: SpartanClaim,
    commitment: Array,
) -> Transcript:
    # The dense matrices are the index in this prototype. An indexed Spartan
    # replaces these frames with a verifier-key digest.
    instance = claim.instance
    transcript = _observe_framed(transcript, 1, instance.a)
    transcript = _observe_framed(transcript, 2, instance.b)
    transcript = _observe_framed(transcript, 3, instance.c)
    transcript = _observe_framed(
        transcript, 4, fnp.array([instance.num_io], instance.a.dtype)
    )
    transcript = _observe_framed(transcript, 5, commitment)
    return _observe_framed(transcript, 6, claim.public_inputs)


class SpartanProver(ProverStage[SpartanClaim, SpartanWitness, None, SpartanProof]):
    """The Spartan prover role; owns the PCS proving capability only."""

    def __init__(
        self,
        pcs_prover: PcsProver[Any, Any, Any],
        *,
        outer: (
            ProverStage[
                ZerocheckClaim, ZerocheckWitness, RowEvaluationClaim, OuterProof
            ]
            | None
        ) = None,
        inner: (
            ProverStage[
                LincheckClaim, LincheckWitness, ColumnEvaluationClaim, InnerProof
            ]
            | None
        ) = None,
        witness_open: (
            ProverStage[
                WitnessOpeningClaim, WitnessOpeningWitness, None, WitnessOpenProof
            ]
            | None
        ) = None,
        challenges: ChallengePolicy = DEFAULT_CHALLENGES,
    ) -> None:
        self.challenges = challenges
        self.pcs_prover = pcs_prover
        self.outer = outer or OuterProver(challenges=challenges)
        self.inner = inner or InnerProver(challenges=challenges)
        self.witness_open = witness_open or WitnessOpenProver(pcs_prover)

    def prove(
        self,
        claim: SpartanClaim,
        witness: SpartanWitness,
        transcript: Transcript,
    ) -> ProveResult[None, SpartanProof]:
        instance = claim.instance
        assignment = witness.assignment
        if assignment.shape != (instance.num_cols,):
            raise ValueError(
                f"expected assignment shape {(instance.num_cols,)}, "
                f"got {assignment.shape}"
            )
        witness_poly = assignment[: instance.num_vars_padded]
        commitment, prover_data = self.pcs_prover.commit([witness_poly])
        transcript = _absorb_claim(transcript, claim, commitment)

        az, bz, cz = instance.matvecs(assignment)
        outer = self.outer.prove(
            ZerocheckClaim(instance.s_x),
            ZerocheckWitness(az, bz, cz),
            transcript,
        )
        batch, transcript = batch_claims(
            outer.reduced_claim.values, outer.transcript, self.challenges
        )
        inner = self.inner.prove(
            LincheckClaim(instance, outer.reduced_claim, batch),
            LincheckWitness(assignment),
            transcript,
        )
        opening_claim = witness_opening_claim(
            commitment,
            instance,
            claim.public_inputs,
            outer.reduced_claim,
            batch,
            inner.reduced_claim,
        )
        opening = self.witness_open.prove(
            opening_claim,
            WitnessOpeningWitness(prover_data),
            inner.transcript,
        )
        return ProveResult(
            None,
            SpartanProof(
                commitment,
                outer.reduction_proof,
                inner.reduction_proof,
                opening.reduction_proof,
            ),
            opening.transcript,
        )


class SpartanVerifier(VerifierStage[SpartanClaim, None, SpartanProof]):
    """The Spartan verifier role; owns the PCS verification capability only."""

    def __init__(
        self,
        pcs_verifier: PcsVerifier[Any, Any],
        *,
        outer: (
            VerifierStage[ZerocheckClaim, RowEvaluationClaim, OuterProof] | None
        ) = None,
        inner: (
            VerifierStage[LincheckClaim, ColumnEvaluationClaim, InnerProof] | None
        ) = None,
        witness_open: (
            VerifierStage[WitnessOpeningClaim, None, WitnessOpenProof] | None
        ) = None,
        challenges: ChallengePolicy = DEFAULT_CHALLENGES,
    ) -> None:
        self.challenges = challenges
        self.outer = outer or OuterVerifier(challenges=challenges)
        self.inner = inner or InnerVerifier(challenges=challenges)
        self.witness_open = witness_open or WitnessOpenVerifier(pcs_verifier)

    def verify(
        self,
        claim: SpartanClaim,
        reduction_proof: SpartanProof,
        transcript: Transcript,
    ) -> VerifyResult[None]:
        transcript = _absorb_claim(transcript, claim, reduction_proof.commitment)
        outer = self.outer.verify(
            ZerocheckClaim(claim.instance.s_x),
            reduction_proof.outer,
            transcript,
        )
        batch, transcript = batch_claims(
            outer.reduced_claim.values, outer.transcript, self.challenges
        )
        inner = self.inner.verify(
            LincheckClaim(claim.instance, outer.reduced_claim, batch),
            reduction_proof.inner,
            transcript,
        )
        opening = self.witness_open.verify(
            witness_opening_claim(
                reduction_proof.commitment,
                claim.instance,
                claim.public_inputs,
                outer.reduced_claim,
                batch,
                inner.reduced_claim,
            ),
            reduction_proof.witness_open,
            inner.transcript,
        )
        return VerifyResult(None, opening.transcript, outer.ok & inner.ok & opening.ok)
