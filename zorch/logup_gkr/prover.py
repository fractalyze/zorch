# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR prover.

`LogupSummand` is the LogUp combine `eq * (lam*(n0*d1 + n1*d0) + d0*d1)` over
five MLE factors `[eq, n0, d1, n1, d0]`, shared by this module's value-form
round and `jagged_prover`'s coeff-form round. `LogupSumcheckRound` is one
per-variable sumcheck round whose summand (`_combine`) delegates to a
`LogupSummand` scoped to its `lam` -- the sibling of the product
`zorch.sumcheck.prover.StandardRound`. Its `__call__` emits a
`RoundMsg(round_poly, challenge)` per round; `GkrLayerRound` stacks the per-round
messages, so the evaluation point is the stacked challenges.

`GkrLayerRound` is one GKR layer: it runs the layer's per-variable LogUp sumcheck
(`fold_rounds` over `LogupSumcheckRound`), then reduces the numerator and
denominator claims across the child selector. The whole GKR
prover is `ProveChain([GkrLayerRound(l) for l in reversed(layers[:-1])])` -- the
interaction floor outward to the input, one bound variable per layer.

The carry threaded through the chain is `(num_eval, den_eval, eval_point)`. The
points follow the MSB-first convention of `zorch.poly.eq` (the sumcheck binds the
high variable first and the eq factor is MSB-indexed, so no reordering is needed;
the pyramid's child selector is the low bit, so it is appended as the last
coordinate). `bind_output` is the shared head, reused by the verifier so their
Fiat-Shamir transcripts cannot diverge; `logup_combine` is module-level so this
round and the verifier oracle evaluate the same expression.

Layers are folded and proved eagerly (`build_pyramid` is a Python loop), not one
fused program: the pyramid does not fit one `@jit` at scale. Scheme-agnostic --
no interaction model, jagged layout, or trace openings; those are the consumer's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr.circuit import GkrLayer, LogUpGkrOutput
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import fold, natural_domain, split_pairs, summand_evals
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.round import ProverRound
    from zorch.sumcheck.gruen import GruenSummand
    from zorch.sumcheck.prover import SumcheckSummand


def logup_combine(
    lam: Array, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array
) -> Array:
    """The LogUp summand `eq * (lam*(n0*d1 + n1*d0) + d0*d1)`.

    Module-level so `LogupSumcheckRound` here and the GKR verifier's oracle
    evaluate the *same* expression -- a drift between them would break soundness
    silently.
    """
    return eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)


class LogupSummand:
    """The LogUp sumcheck summand shared by the dense round (value-form) and the
    jagged round (coeff-form): the combine `eq*(lam*(n0*d1+n1*d0)+d0*d1)` over
    [eq, n0, d1, n1, d0], its loop-invariant scalar (lam), and its degree. Single
    source of the combine math the verifier oracle also calls. For the jagged
    engine it also owns the two GruenSummand slots — `paired_evals` (the
    materialized evaluations at the summand's points) and `correct` (the
    padding correction) — so a jagged round body reads evaluate → correct →
    assemble with only this class carrying protocol content."""

    DEGREE: int = 3  # eq (deg1) * (lam*cross + d0*d1) (deg2)
    NUM_FACTORS: int = 5  # [eq, n0, d1, n1, d0]

    def __init__(self, lam: Array) -> None:
        self.lam = lam

    @property
    def degree(self) -> int:
        return self.DEGREE

    def combine_scalars(self) -> tuple[Array, ...]:
        return (self.lam,)

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        if len(factors) != self.NUM_FACTORS:
            raise ValueError(
                f"LogUp combine needs {self.NUM_FACTORS} factors [eq, n0, d1, n1, d0], "
                f"got {len(factors)}"
            )
        (lam,) = scalars
        return logup_combine(lam, *factors)

    def _combine(self, *factors: Array) -> Array:
        """LogUp summand bound to its scalars (λ); the `_combine` seam a generic
        round-poly builder reads, so `StandardRound` folds LogUp directly."""
        return self.combine(self.combine_scalars(), *factors)

    @classmethod
    def extra_ts(cls, dtype: Any) -> tuple[Array, ...]:
        """The jagged round's one materialized extra point ``{1/2}`` — degree
        3 makes d - 2 = 1 extra beyond s(0), the Gruen seam's invariant
        (`zorch.sumcheck.gruen.GruenSummand`). The dense (value-form) round
        never reads this."""
        one = jnp.ones((), dtype)
        return (one / jnp.array(2, dtype),)

    def paired_evals(
        self, n0: Array, n1: Array, d0: Array, d1: Array, eq_0: Array, eq_1: Array
    ) -> tuple[Array, Array, Array]:
        """The jagged round's materialized evaluations — the GruenSummand
        evals slot: ``(s(0), 8*s(1/2), eq mass)`` over the stride-2 pairs.

        s(0) reads the even elements at their eq weight; the u=1/2 sum works
        on doubled values (``e0 + e1 = 2*e(1/2)`` per factor, likewise eq) so
        no division enters the kernel — `correct` rescales. Both points go
        through `combine` so the summand cannot drift from the verifier
        oracle's."""
        scalars = self.combine_scalars()
        (n0_0, n0_1), (n1_0, n1_1) = split_pairs(n0), split_pairs(n1)
        (d0_0, d0_1), (d1_0, d1_1) = split_pairs(d0), split_pairs(d1)
        eval_zero = jnp.sum(self.combine(scalars, eq_0, n0_0, d1_0, n1_0, d0_0))
        eq_h = eq_0 + eq_1
        eval_half = jnp.sum(
            self.combine(
                scalars,
                eq_h,
                n0_0 + n0_1,
                d1_0 + d1_1,
                n1_0 + n1_1,
                d0_0 + d0_1,
            )
        )
        return eval_zero, eval_half, jnp.sum(eq_h)

    def correct(
        self,
        eval_zero: Array,
        eval_half: Array,
        eq_sum: Array,
        eq_adj: Array,
        pad_adj: Array,
        z_cur: Array,
    ) -> tuple[Array, Array]:
        """The padding correction — the GruenSummand correction slot: add the
        non-materialized positions' contribution back in closed form.

        Every virtual position holds the fold-neutral fraction (n=0, d=1),
        whose summand collapses to just its eq weight; the eq weights of the
        full remaining hypercube sum to ``pad_adj``, so the virtual mass is
        ``pad_adj - eq_sum``. At u=0 it carries the current variable's eq
        factor ``(1 - z_cur)``; at the doubled u=1/2 scale each virtual pair
        contributes ``den_h = 4`` at weight ``eq_h = eq_rest``, and the
        doubled products overcount s(1/2) by 8. ``eq_adj`` is the row-eq
        residual scalar once the row variables are exhausted (1 before
        that)."""
        dtype = z_cur.dtype
        one = jnp.ones((), dtype)
        correction = pad_adj - eq_sum
        s_zero = (eval_zero + correction * (one - z_cur)) * eq_adj
        s_half = (
            (eval_half + correction * jnp.array(4, dtype))
            / jnp.array(8, dtype)
            * eq_adj
        )
        return s_zero, s_half


def fold_carry(
    n0: Array, n1: Array, d0: Array, d1: Array, point: Array, r: Array
) -> tuple[Array, Array, Array]:
    """Reduce a LogUp layer's four openings to the next GKR carry under the
    child selector `r`: bind num/den, and append `r` as the low (last) bit of
    the MSB-first point. Module-level for the same reason as `logup_combine` --
    the prover and verifier carry folds must stay byte-identical or the verifier
    accepts proofs the prover stopped producing."""
    num_eval = n0 + (n1 - n0) * r
    den_eval = d0 + (d1 - d0) * r
    return num_eval, den_eval, jnp.concatenate([point, jnp.atleast_1d(r)])


@partial(jax.tree_util.register_dataclass, data_fields=["lam"], meta_fields=[])
@dataclass(frozen=True)
class LogupSumcheckRound(Round):
    """Per-variable sumcheck round for the LogUp combine (sibling of the product
    `zorch.sumcheck.prover.StandardRound`); emits a `RoundMsg`."""

    # Batching challenge; fixed across a layer's variable-rounds.
    lam: Array

    @property
    def _summand(self) -> LogupSummand:
        """The shared LogUp combine, scoped to this round's lam. A derived
        property, not a dataclass field, so `lam` stays the only pytree leaf --
        the registered dataclass above only knows about `lam`."""
        return LogupSummand(self.lam)

    @property
    def degree(self) -> int:
        return LogupSummand.DEGREE

    def combine_scalars(self) -> tuple[Array, ...]:
        """The batching challenge λ, fixed across the layer's variable-rounds; the
        marked path threads it as a marker operand so a vendor feeds the inlined
        combine."""
        return (self.lam,)

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        """LogUp summand over [eq, n0, d1, n1, d0] (the scalar-explicit seam):
        single source of the combine math -- `_combine`, the round-poly reduction,
        and the marked path's nested combine region all route here. Delegates to
        the shared `LogupSummand`, which itself calls the module-level
        `logup_combine` the verifier oracle also calls, so prover and verifier
        cannot drift. `LogupSummand` guards the factor count at this summand seam
        -- both `_round_poly` and the scan driver reach it, so neither rechecks
        (arg count is static, so the guard is trace-safe)."""
        return self._summand.combine(scalars, *factors)

    def _combine(self, *factors: Array) -> Array:
        """LogUp summand bound to its scalars (λ)."""
        return self.combine(self.combine_scalars(), *factors)

    def _round_poly(self, folded: Array) -> Array:
        """Round polynomial over the natural {0..degree} evals, shape
        (degree+1, *batch): one batched `summand_evals` reduction of the LogUp
        combine over the stacked [eq, n0, d1, n1, d0] factors."""
        return summand_evals(
            folded, self._combine, natural_domain(self.degree, folded.dtype)
        )

    def __call__(
        self, folded: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, RoundMsg]:
        msg = self._round_poly(folded)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return fold(folded, r[0]), transcript, RoundMsg(msg, r[0])


@dataclass(frozen=True)
class LayerProof:
    """One GKR layer's sumcheck transcript: round polynomials, the bound
    point, and the final openings.

    `point` exists for wire serialization: a consumer emitting per-layer
    (point, openings) records reads it here instead of replaying the
    transcript or peeking at the chain carry's layout. A verifier derives its
    own point from the transcript replay and must never read this field — it
    is prover-asserted, not transcript-bound."""

    round_polys: Array  # (num_variables, degree + 1), each round's univariate
    point: Array  # the bound point, MSB-first (the sampled challenges in order)
    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array


Carry = tuple[Array, Array, Array]  # (num_eval, den_eval, eval_point)


def bind_output(
    output: LogUpGkrOutput, transcript: Transcript
) -> tuple[Carry, Transcript]:
    """Commit the circuit output and draw the initial evaluation claim.

    The shared head of both chains: observe the output numerator/denominator,
    sample a point over their variables, and evaluate. Returns the initial carry
    `(num_eval, den_eval, eval_point)` and the advanced transcript.
    """
    num_vars = log2_strict_usize(output.numerator.shape[0])
    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(output.denominator)
    transcript, eval_point = transcript.sample(num_vars)
    num_eval = eval_mle(output.numerator, eval_point)
    den_eval = eval_mle(output.denominator, eval_point)
    return (num_eval, den_eval, eval_point), transcript


class GkrLayerRound(Round):
    """Prove one GKR layer; the chain of these (floor outward) is the GKR prover."""

    def __init__(self, layer: GkrLayer) -> None:
        self.layer = layer

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, LayerProof]:
        num_eval, den_eval, eval_point = carry
        transcript, lam = transcript.sample(1)
        lam = lam[0]
        one = jnp.ones((), eval_point.dtype)
        # State order is LogupSumcheckRound's: [eq, n0, d1, n1, d0], stacked (5, N).
        state = jnp.stack(
            [
                expand_eq_to_hypercube(eval_point, one),
                self.layer.numerator_0,
                self.layer.denominator_1,
                self.layer.numerator_1,
                self.layer.denominator_0,
            ]
        )
        rounds = log2_strict_usize(state.shape[-1])
        final_state, transcript, msgs = fold_rounds(
            LogupSumcheckRound(lam), state, transcript, rounds
        )
        round_polys = jnp.stack([m.round_poly for m in msgs])
        point = jnp.stack([m.challenge for m in msgs])

        _, n0, d1, n1, d0 = final_state[:, 0]
        transcript, r = transcript.observe_and_sample(jnp.stack([n0, n1, d0, d1]), 1)
        r = r[0]
        num_eval, den_eval, eval_point = fold_carry(n0, n1, d0, d1, point, r)

        proof = LayerProof(round_polys, point, n0, n1, d0, d1)
        return (num_eval, den_eval, eval_point), transcript, proof


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _summand: type[SumcheckSummand] = LogupSumcheckRound
    _summand_value: type[SumcheckSummand] = LogupSummand
    _gruen_summand: type[GruenSummand] = LogupSummand
    _sumcheck_round: type[ProverRound] = LogupSumcheckRound
    _layer_round: type[ProverRound] = GkrLayerRound
