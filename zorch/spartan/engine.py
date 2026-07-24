# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-variable sumcheck engine injected into a Spartan stage.

Each stage (zerocheck / lincheck) runs a sumcheck over its operand; `StageSumcheck`
is the matched **(prover round, verifier round)** pair that runs it. Injecting a
different pair swaps the sumcheck algorithm, evaluation domain, or wire form
without touching the stage's schedule or its terminal identity check — e.g. the
compressed coefficient wire (`CompressedProductRound` ↔
`CompressedCoeffsSumcheckRound`), a √-space or eq-engine prover, or an
extension-field challenge domain.

The two rounds must be duals: they observe the round message and sample the
challenge identically, or prover and verifier desync. The prover stage also
recovers its bound point by *replaying the verifier round* over the proof
(`zorch.verify`), so a consistent pair keeps the point wire-agnostic.
The verifier round defines how the stage interprets the message form.

The default engines reproduce the shipped behavior: value-form `StandardRound`
over the degree-3 zerocheck summand `eq·(â◦b̂−ĉ)` (natural domain) for the outer
stage, and the degree-2 product for the inner. The outer summand is mixed-degree,
so its domain must stay finite (a leading-∞ domain needs a homogeneous summand);
a swapped outer engine must respect that.
"""

from __future__ import annotations

from dataclasses import dataclass

from zorch.round import InnerVerifierRound, ProverRound
from zorch.spartan.summand import ZerocheckSummand
from zorch.sumcheck.prover import ProductSummand, StandardRound
from zorch.sumcheck.verifier import SumcheckRound

ZEROCHECK_DEGREE = 3
LINCHECK_DEGREE = 2


@dataclass(frozen=True)
class StageSumcheck:
    """A matched per-variable sumcheck-round pair for a Spartan stage — the prover
    round that folds the operand and the verifier round that is its dual."""

    prover_round: ProverRound
    verifier_round: InnerVerifierRound


def zerocheck_engine() -> StageSumcheck:
    """Default outer engine: value-form product `StandardRound` over the degree-3
    zerocheck summand `eq·(â◦b̂−ĉ)` on the natural domain."""
    return StageSumcheck(
        StandardRound(ZerocheckSummand()), SumcheckRound(ZEROCHECK_DEGREE)
    )


def lincheck_engine() -> StageSumcheck:
    """Default inner engine: value-form degree-2 product `StandardRound`."""
    return StageSumcheck(
        StandardRound(ProductSummand(LINCHECK_DEGREE)), SumcheckRound(LINCHECK_DEGREE)
    )
