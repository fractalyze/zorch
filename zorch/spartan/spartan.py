# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The thin Spartan assembly: wire the R1CS combinators into a proof.

The "scheme" layer — it owns only the schedule and Fiat-Shamir framing,
composing the agnostic combinators into a `ProveChain` / `VerifyChain`. A
different R1CS-proving schedule reuses the same combinators under a different
assembly; the PCS is injected (any `zorch.pcs.protocol` pair).

Schedule (prover and verifier identical): commit `W`, absorb the commitment +
public inputs, then run `[Outer, RLC, Inner, WitnessOpen]`. The four messages
plus the commitment are the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import Array

from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.round import ProveChain, VerifyChain
from zorch.spartan.carry import SpartanCarry
from zorch.spartan.engine import StageSumcheck
from zorch.spartan.lincheck import (
    InnerProver,
    InnerVerifier,
    RlcProver,
    RlcVerifier,
)
from zorch.spartan.pcs_glue import WitnessOpenProver, WitnessOpenVerifier
from zorch.spartan.r1cs import R1CS
from zorch.spartan.zerocheck import OuterProver, OuterVerifier
from zorch.transcript import Transcript


@dataclass(frozen=True)
class SpartanProof:
    """A Spartan proof: the witness commitment plus the chain's per-stage messages
    (outer `(round_polys, claims)`, RLC `None`, inner `round_polys`, open
    `(values, proof)`)."""

    commitment: Array
    messages: list[Any]


def _absorb_statement(
    transcript: Transcript, commitment: Array, io: Array
) -> Transcript:
    """Bind the commitment and public inputs before sampling any challenge — the
    statement the proof is about. Prover and verifier absorb identically."""
    transcript = transcript.observe(commitment)
    if io.shape[0] > 0:
        transcript = transcript.observe(io)
    return transcript


def prove(
    instance: R1CS,
    z: Array,
    io: Array,
    pcs_prover: PcsProver[Any, Any, Any],
    transcript: Transcript,
    *,
    outer: StageSumcheck | None = None,
    inner: StageSumcheck | None = None,
) -> tuple[SpartanProof, Transcript]:
    """Prove `(A·z)∘(B·z) = C·z` for the witness-first assignment `z = (W,1,X)`.

    `outer` / `inner` swap the zerocheck / lincheck sumcheck engine; pass the same
    pair to `verify`.
    """
    az, bz, cz = instance.matvecs(z)
    witness = z[: instance.num_vars_padded]
    commitment, prover_data = pcs_prover.commit([witness])
    transcript = _absorb_statement(transcript, commitment, io)
    chain = ProveChain(
        [
            OuterProver(az, bz, cz, sumcheck=outer),
            RlcProver(),
            InnerProver(instance, z, sumcheck=inner),
            WitnessOpenProver(pcs_prover, prover_data),
        ]
    )
    _, transcript, messages = chain(SpartanCarry(), transcript)
    return SpartanProof(commitment, messages), transcript


def verify(
    instance: R1CS,
    io: Array,
    proof: SpartanProof,
    pcs_verifier: PcsVerifier[Any, Any],
    transcript: Transcript,
    *,
    outer: StageSumcheck | None = None,
    inner: StageSumcheck | None = None,
) -> tuple[Array, Transcript]:
    """Verify a `SpartanProof`; returns `(ok, transcript)`. `outer` / `inner` must
    match the engines passed to `prove`."""
    transcript = _absorb_statement(transcript, proof.commitment, io)
    chain = VerifyChain(
        [
            OuterVerifier(sumcheck=outer),
            RlcVerifier(),
            InnerVerifier(sumcheck=inner),
            WitnessOpenVerifier(pcs_verifier, proof.commitment, instance, io),
        ]
    )
    _, transcript, ok = chain(SpartanCarry(), proof.messages, transcript)
    return ok, transcript
