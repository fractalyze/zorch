# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Spartan as an explicit composition of conditional claim reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.spartan.lincheck import (
    InnerProof,
    InnerStage,
    LincheckClaim,
    LincheckWitness,
    batch_claims,
)
from zorch.spartan.pcs_glue import (
    WitnessOpeningWitness,
    WitnessOpenProof,
    WitnessOpenStage,
    witness_opening_claim,
)
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import (
    OuterProof,
    OuterStage,
    ZerocheckClaim,
    ZerocheckWitness,
)
from zorch.stage import ProveResult, Stage, VerifyResult
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


class Spartan(Stage[SpartanClaim, SpartanWitness, None, SpartanProof]):
    """Close a Spartan satisfiability claim through three child reductions."""

    def __init__(
        self,
        pcs_prover: PcsProver[Any, Any, Any],
        pcs_verifier: PcsVerifier[Any, Any],
        *,
        outer: OuterStage | None = None,
        inner: InnerStage | None = None,
        witness_open: WitnessOpenStage | None = None,
        challenges: ChallengePolicy | None = None,
    ) -> None:
        self.challenges = challenges or ChallengePolicy()
        self.pcs_prover = pcs_prover
        self.outer = outer or OuterStage(challenges=self.challenges)
        self.inner = inner or InnerStage(challenges=self.challenges)
        self.witness_open = witness_open or WitnessOpenStage(pcs_prover, pcs_verifier)

    @staticmethod
    def _observe_framed(transcript: Transcript, tag: int, values: Array) -> Transcript:
        header = fnp.array([tag, values.ndim, *values.shape], values.dtype)
        transcript = transcript.observe(header)
        if values.size > 0:
            transcript = transcript.observe(fnp.reshape(values, (-1,)))
        return transcript

    @classmethod
    def _absorb_claim(
        cls,
        transcript: Transcript,
        claim: SpartanClaim,
        commitment: Array,
    ) -> Transcript:
        # The dense matrices are the index in this prototype. An indexed Spartan
        # replaces these three frames with the verifier-key digest, preserving the
        # same binding contract without absorbing the full matrices per proof.
        instance = claim.instance
        transcript = cls._observe_framed(transcript, 1, instance.a)
        transcript = cls._observe_framed(transcript, 2, instance.b)
        transcript = cls._observe_framed(transcript, 3, instance.c)
        transcript = cls._observe_framed(
            transcript,
            4,
            fnp.array([instance.num_io], instance.a.dtype),
        )
        transcript = cls._observe_framed(transcript, 5, commitment)
        return cls._observe_framed(transcript, 6, claim.public_inputs)

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
        transcript = self._absorb_claim(transcript, claim, commitment)

        az, bz, cz = instance.matvecs(assignment)
        outer = self.outer.prove(
            ZerocheckClaim(instance.s_x),
            ZerocheckWitness(az, bz, cz),
            transcript,
        )
        batch, transcript = batch_claims(
            outer.reduced_claim.values, outer.transcript, self.challenges
        )
        lincheck_claim = LincheckClaim(instance, outer.reduced_claim, batch)
        inner = self.inner.prove(
            lincheck_claim,
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
        reduction_proof = SpartanProof(
            commitment,
            outer.reduction_proof,
            inner.reduction_proof,
            opening.reduction_proof,
        )
        return ProveResult(None, reduction_proof, opening.transcript)

    def verify(
        self,
        claim: SpartanClaim,
        reduction_proof: SpartanProof,
        transcript: Transcript,
    ) -> VerifyResult[None]:
        transcript = self._absorb_claim(transcript, claim, reduction_proof.commitment)
        outer = self.outer.verify(
            ZerocheckClaim(claim.instance.s_x),
            reduction_proof.outer,
            transcript,
        )
        batch, transcript = batch_claims(
            outer.reduced_claim.values, outer.transcript, self.challenges
        )
        lincheck_claim = LincheckClaim(claim.instance, outer.reduced_claim, batch)
        inner = self.inner.verify(
            lincheck_claim,
            reduction_proof.inner,
            transcript,
        )
        opening_claim = witness_opening_claim(
            reduction_proof.commitment,
            claim.instance,
            claim.public_inputs,
            outer.reduced_claim,
            batch,
            inner.reduced_claim,
        )
        opening = self.witness_open.verify(
            opening_claim,
            reduction_proof.witness_open,
            inner.transcript,
        )
        ok = outer.ok & inner.ok & opening.ok
        return VerifyResult(None, opening.transcript, ok)
