# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck verifier rounds and their scan driver -- the per-variable duals of
the prover family.

`SumcheckRound` checks the round-poly identity and reduces the claim. It is
summand-agnostic: it sees only the round polynomials, so one verifier serves
every prover summand (product, LogUp, ...) at a given `degree`. The
observe -> sample order matches `prover.SumcheckRound.__call__` exactly, so
the prover's and verifier's Fiat-Shamir transcripts cannot diverge.

`CoeffsSumcheckRound` is the same check for a prover that sends ascending
coefficients instead of natural-domain values -- the wire form of a round
interpolated off a non-natural node set (e.g. through an eq factor's root),
where value form would force the verifier to know the sender's nodes. It also
owns the challenge squeeze rule (`challenge_limbs`), since a coefficient
prover's claims may live in an extension of the transcript's field.

`verify` scans either round over a proof -- the dual of `prover.prove`, and one
`lax.scan` for the same reason: the replay compiles to a single traced region
flat in the round count (issue #58) rather than an unrolled body that crosses the
ZKX PTX cliff.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.poly.univariate import eval_coeffs, eval_univariate
from zorch.round import Round
from zorch.transcript import Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.round import InnerVerifierRound


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["degree"])
@dataclass(frozen=True)
class SumcheckRound(Round):
    """Verifier for any sumcheck round; the dual of `prover.SumcheckRound`."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} evals, "
                f"got {msg.shape[0]}"
            )
        ok = claim == msg[0] + msg[1]
        transcript, r = transcript.observe_and_sample(msg, 1)
        return eval_univariate(msg, r[0]), transcript, r[0], ok


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["degree", "challenge_limbs"],
)
@dataclass(frozen=True)
class CoeffsSumcheckRound(Round):
    """Verifier for a coefficient-form sumcheck round: `s(0) = c_0` and
    `s(1) = sum(c)`, so the identity check and the claim reduction read the
    coefficients directly."""

    degree: int
    challenge_limbs: int = 1

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.challenge_limbs < 1:
            raise ValueError("challenge_limbs must be >= 1")

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != self.degree + 1:
            raise ValueError(
                f"round message must have degree+1={self.degree + 1} "
                f"coefficients, got {msg.shape[0]}"
            )
        ok = claim == msg[0] + jnp.sum(msg)
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, claim.dtype, self.challenge_limbs)
        return eval_coeffs(msg, r), transcript, r, ok


def verify(
    verifier: InnerVerifierRound, claim: Array, proof: Array, transcript: Transcript
) -> tuple[Array, Transcript, Array, Array]:
    """Replay `proof` against `claim` → `(final_claim, transcript, point, ok)`.

    `point` is the evaluation point the rounds bound; `ok` ANDs every round's
    check, so one false anywhere rejects the proof. The reduced `(final_claim,
    point)` is where the verifier stops — closing it needs a PCS opening, which
    is the consumer's.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")

    def step(
        carry: tuple[Array, Transcript], msg: Array
    ) -> tuple[tuple[Array, Transcript], tuple[Array, Array]]:
        claim, transcript = carry
        claim, transcript, r, ok = verifier(claim, msg, transcript)
        return (claim, transcript), (r, ok)

    (claim, transcript), (point, oks) = lax.scan(step, (claim, transcript), proof)
    return claim, transcript, point, jnp.all(oks)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _eval_form: type[InnerVerifierRound] = SumcheckRound
    _coeffs_form: type[InnerVerifierRound] = CoeffsSumcheckRound
