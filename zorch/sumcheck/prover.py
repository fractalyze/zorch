# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck prover rounds: the product `StandardRound`, its compressed-wire
sibling `CompressedProductRound`, and the summand seam they read.

A sumcheck round splits each MLE on the current variable, sends the round
polynomial sampled over an `EvalDomain`, then folds every MLE at the verifier's
challenge (P0 + r*(P1 - P0)). Split and fold are summand-independent, and the
round poly is built by `zorch.sumcheck.domain.summand_evals` over the stacked
`(m, N)` state -- generic over BOTH the summand (`SumcheckSummand._combine`:
product, LogUp, ...) and the sampling domain. `StandardRound` (here) is the plain
materialized round: it holds the full factor table and does split -> sample ->
combine -> sum, the linear-time reference the memory-optimized siblings
(`sqrt_space.SqrtSpaceRound`, `eq.SmallValueRound`) specialize. Its summand
defaults to the product (`ProductSummand`) and its domain to the natural
{0..degree} evals.

The dense round binds MSB-first (`domain.fold`); the jagged engines bind
LSB-first (`domain.fold(..., msb=False)`). The split/fold primitives and the
round-poly builder (`summand_evals`) live in `zorch.sumcheck.domain`; the
verifier dual in `zorch.sumcheck.verifier`.

Rounds run under the scheme-agnostic `zorch.prove.fold_rounds` host loop (any
`Round`, any message shape) -- one round per variable, folding the state down
each step. `SumcheckSummand` (`degree` + `_combine`) is the summand seam the
round-poly builder reads, so the product `ProductSummand` and the LogUp
`logup_gkr.prover.LogupSummand` drive it interchangeably.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Protocol

import frx
import frx.numpy as fnp
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.round import ProverRound, RunningClaim
from zorch.sumcheck.domain import (
    EvalDomain,
    fold,
    natural_domain,
    summand_evals,
)
from zorch.sumcheck.reduce import reduce_compressed, reduce_domain, reduce_evals
from zorch.transcript import Transcript


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["state", "claim"],
    meta_fields=[],
)
@dataclass(frozen=True)
class FoldingClaim:
    """What a sumcheck round threads: the engine's folding state and the claim
    built so far.

    `state` is whatever the engine folds — stacked factor tables for the
    materialized rounds, a deferred `(factors, eq)` pair for the √-space one.

    The claim half is why the prover can export a reduced claim at all. A round
    squeezes its challenge, folds with it, and would otherwise drop it — and the
    point and running value live nowhere else. Carrying them is the round
    contract\'s own rule, that derived state belongs in the carry, applied to the
    role that derives them first.
    """

    state: Any
    claim: RunningClaim

    def advance(self, state: Any, reduced: Array, challenge: Array) -> FoldingClaim:
        """The folded state beside the claim this round reduced it to."""
        return FoldingClaim(state, self.claim.bind(reduced, challenge))


@partial(frx.jit, static_argnames=("degree",))
def _fold_and_advance(
    carry: FoldingClaim, msg: Array, r: Array, degree: int
) -> FoldingClaim:
    """Fold and reduce in one compiled step, so the claim's handful of field ops
    ride the fold's program rather than paying for a launch of their own."""
    reduced, _ = reduce_evals(carry.claim.value, msg, r, degree)
    return carry.advance(fold(carry.state, r), reduced, r)


def initial_claim(state: Any, value: Array, rounds: int) -> FoldingClaim:
    """Start a fold from `state` with nothing yet bound into `value`."""
    return FoldingClaim(
        state, RunningClaim(value, fnp.zeros((rounds,), value.dtype), fnp.int32(0))
    )


if TYPE_CHECKING:
    from zorch.round import ProverRound, RunningClaim


@dataclass(frozen=True)
class ProductSummand:
    """The product sumcheck summand `s = Σ_x Πₖ Pₖ(x)`: the combine math alone, no
    round machinery. The default summand of `StandardRound`; the `eq` / `sqrt_space`
    engines hold one to weight their eq-product. Pairs with the LogUp
    `logup_gkr.prover.LogupSummand` under the `SumcheckSummand` seam."""

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

    def combine_scalars(self) -> tuple[Array, ...]:
        """No loop-invariant scalars: the product reads only its factors."""
        return ()

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        """Product `Πₖ fₖ` (the scalar-explicit seam; product takes no scalars).
        Single source of the combine math: `_combine` and any marked path route
        here, so they cannot drift."""
        del scalars  # product has none
        return reduce(operator.mul, factors)

    def _combine(self, *factors: Array) -> Array:
        """Product bound to its (empty) scalars; the round-poly builder reads only
        this, so callers stay summand-generic."""
        return self.combine(self.combine_scalars(), *factors)


class StandardRound(ProverRound):
    """The plain materialized sumcheck round: send the summand's round poly over
    `domain`, sample the challenge, fold the stacked `(m, N)` state. Bound to a
    `SumcheckSummand` (product by default) and an `EvalDomain` (the natural
    {0..degree} evals when None). This is the linear-time reference the √-space /
    eq engines specialize; driven by `fold_rounds`.

    `challenges` names the field challenges are drawn in. Naming the
    transcript's own field is the one-squeeze schedule; an extension policy lets
    a tail whose earlier round bound in that extension continue folding in the
    same field."""

    def __init__(
        self,
        summand: SumcheckSummand,
        domain: EvalDomain | None = None,
        *,
        challenges: ChallengePolicy,
    ) -> None:
        self.summand = summand
        self.domain = domain
        self.challenges = challenges

    def _round_poly(self, folded: Array) -> Array:
        """s sampled at `domain` (the natural {0..degree} evals by default), shape
        (num_points, *batch): one batched `summand_evals` reduction over the stacked
        factors, so it lowers toward a single reduction kernel."""
        domain = self.domain or natural_domain(self.summand.degree, folded.dtype)
        return summand_evals(folded, self.summand._combine, domain)

    def __call__(
        self, carry: FoldingClaim, transcript: Transcript
    ) -> tuple[FoldingClaim, Transcript, Array]:
        domain = self.domain or natural_domain(self.summand.degree, carry.state.dtype)
        msg = summand_evals(carry.state, self.summand._combine, domain)
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        if self.domain is None:
            return (
                _fold_and_advance(carry, msg, r, self.summand.degree),
                transcript,
                msg,
            )
        # The default domain reduces by direct Lagrange evaluation — the same
        # arithmetic `verifier.SumcheckRound` does. Going through the domain's
        # value→coefficient map instead would rebuild an (n, n) Lagrange matrix
        # every round, which costs more than the fold it rides along with.
        reduced, _ = reduce_domain(carry.claim.value, msg, r, domain)
        return carry.advance(fold(carry.state, r), reduced, r), transcript, msg


class CompressedProductRound(ProverRound):
    """Two-factor product round with the compressed coefficient wire: the message
    is `[c_0, c_2]` — the degree-2 round polynomial's constant and leading
    coefficients — and the linear coefficient stays off the wire (the verifier
    dual, `verifier.CompressedCoeffsSumcheckRound`, reconstructs it from the
    running claim via `s(0) + s(1) = claim`). Split and fold match
    `StandardRound(ProductSummand(2))` exactly (the MSB variable binds); only the
    message form differs, so a scheme whose wire fixes this form swaps rounds
    without touching the fold. `c_2 = Σ (P1_f - P0_f)·(P1_b - P0_b)` is the honest
    leading coefficient in any characteristic; over char 2 it coincides with the
    `(P0 + P1)` products some wire specs write it as."""

    def __init__(self, challenges: ChallengePolicy) -> None:
        self.challenges = challenges

    def _round_poly(self, folded: Array) -> Array:
        """`[c_0, c_2]` of `s(X) = Σ_x' f(X, x')·b(X, x')`, shape (2, *batch):
        one stacked element-wise product per coefficient, then the single
        inherent Σ. The two stacked factors split MSB-first into (f0, f1), (b0, b1)."""
        if folded.shape[0] != 2:
            raise ValueError(
                f"compressed product round takes exactly 2 factors, got "
                f"{folded.shape[0]}"
            )
        (f0, f1), (b0, b1) = fnp.reshape(folded, (2, 2, -1))
        return fnp.sum(fnp.stack([f0 * b0, (f1 - f0) * (b1 - b0)]), axis=-1)

    def __call__(
        self, carry: FoldingClaim, transcript: Transcript
    ) -> tuple[FoldingClaim, Transcript, Array]:
        msg = self._round_poly(carry.state)
        transcript, r = self.challenges.observe_and_sample(transcript, msg)
        reduced, _ = reduce_compressed(carry.claim.value, msg, r)
        return carry.advance(fold(carry.state, r), reduced, r), transcript, msg


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["round_poly", "challenge"],
    meta_fields=[],
)
@dataclass(frozen=True)
class RoundMsg:
    """One per-variable sumcheck round's message: the round polynomial sent plus
    the Fiat-Shamir challenge it induced — the round_polys form the proof and the
    challenges the evaluation point. The challenge is re-derivable from `round_poly`,
    so it never goes on the wire; it rides here only to spare the prover a transcript
    replay. `LogupSumcheckRound.__call__` emits one per round for the `fold_rounds`
    driver. Registered as a pytree because a downstream `lax.scan` stacks it as its
    output — a scan output leaf must be a valid JAX type."""

    round_poly: Array
    challenge: Array


class SumcheckSummand(Protocol):
    """The summand seam a per-variable round exposes: the round-poly `degree`, and
    `_combine` — the summand over the lifted factors. The round-poly builder reads
    only this, so one builder serves every sumcheck; `ProductSummand` (product) and
    `logup_gkr.prover.LogupSummand` (LogUp) both satisfy it, and the host-loop
    engines (sqrt_space / eq) pass a summand to `summand_evals`.

    `degree` is a read-only property here so a frozen-dataclass field (product)
    and a `@property` (LogUp) both match — a plain `degree: int` would demand a
    settable attribute that neither provides.

    `combine` is the scalar-explicit form of the summand and `combine_scalars` the
    loop-invariant scalars it reads (LogUp's λ; empty for product), so they bind once
    rather than per round."""

    @property
    def degree(self) -> int: ...

    def combine_scalars(self) -> tuple[Array, ...]: ...

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array: ...

    def _combine(self, *factors: Array) -> Array: ...


# The FS-less compute-only round marker: the jagged LogUp-GKR host
# loop wraps each round's fold+sum (no Fiat-Shamir) in this composite, while the
# separate `zorch.poseidon2` marker carries FS between rounds. The composite
# attributes the recognizing emitter parses:
#   phase   -- "first" (round 0, no fold), "mid" (fold-by-alpha then sum),
#              "boundary" (the row->interaction handoff: fold-by-alpha then sum
#              over the still-unfolded eq, which rides through un-bound), or
#              "final" (fold only, emitting the four pair openings); routes to
#              the round kernel by position.
#   variant -- the round-kernel shape: "dense" (the uniform interaction round --
#              a batched LogUp-GKR round-poly, folds densely) or "jagged" (the row
#              phase -- segment-based, variable heights, runtime row_counts).
#   degree    -- round-poly degree.
#   poly_form -- the round-poly evaluation domain/form ("coefficient" = the Gruen
#                {0, 1/2} + s(1)=claim-s(0) scheme this round uses).
# No `num_scalars`: the LogUp summand is hardcoded in the emitter (AccumLogupPair),
# which does not read it (it matters only for a generic `zorch.sumcheck.combine`
# region). No `challenge_limbs`: the separate FS step recomposes the fold
# challenge, which arrives as one operand whose dtype already carries base vs
# extension.
SUMCHECK_ROUND_MARKER = "zorch.sumcheck.round"
# Version 1: this marker never shipped, and its producer (here), the XLA
# `SumcheckRecognizer` (`kSumcheckRoundCompositeVersion`), and the emitters are
# pinned together, so the version is the initial one and moves only on a future
# cross-release ABI break. Keep in lockstep with the XLA recognizer's constant.
SUMCHECK_ROUND_MARKER_VERSION = 1


if TYPE_CHECKING:
    _summand: type[SumcheckSummand] = ProductSummand
    _prover_round: type[ProverRound] = StandardRound
    _compressed_prover_round: type[ProverRound] = CompressedProductRound
