# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck verifier round -- the per-variable dual of the prover family.

`SumcheckRound` checks the round-poly identity and reduces the claim. It is
summand-agnostic: it sees only the round polynomials, so one verifier serves
every prover summand (product, LogUp, ...) at a given `degree`. The
observe -> sample order matches `prover.SumcheckRound.__call__` exactly, so
the prover's and verifier's Fiat-Shamir transcripts cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
from jax import Array

from zorch.poly.univariate import eval_univariate
from zorch.round import Round
from zorch.transcript import Transcript


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
