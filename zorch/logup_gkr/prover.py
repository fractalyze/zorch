# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR prover.

`LogupSummand` is the LogUp combine `eq * (lam*(n0*d1 + n1*d0) + d0*d1)` over
five MLE factors `[eq, n0, d1, n1, d0]`, shared by this module's value-form
round and `jagged_prover`'s coeff-form round. `LogupSumcheckRound` is one
per-variable sumcheck round whose summand (`_combine`) delegates to a
`LogupSummand` scoped to its `lam` -- the sibling of the product
`zorch.sumcheck.prover.SumcheckRound`. Its `__call__` emits a
`RoundMsg(round_poly, challenge)` for the generic `fold_rounds` driver; the
homogeneous scan driver `prove` (which `GkrLayerRound` uses) reads only
`degree` + `_combine` and stacks that same `RoundMsg`, so the evaluation point
is `msgs.challenge`.

`GkrLayerRound` is one GKR layer: it runs the layer's per-variable LogUp sumcheck
(via the homogeneous scan driver `prove` over `LogupSumcheckRound`), then reduces
the numerator and denominator claims across the child selector. The whole GKR
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
from zorch.round import Round
from zorch.sumcheck.prover import RoundMsg, factors_on_domain, fold, prove
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
    source of the combine math the verifier oracle also calls."""

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

    @classmethod
    def extra_ts(cls, dtype: Any) -> tuple[Array, ...]:
        """The jagged round's one materialized extra point ``{1/2}`` — degree
        3 makes d - 2 = 1 extra beyond s(0), the Gruen seam's invariant
        (`zorch.sumcheck.gruen.GruenSummand`). The dense (value-form) round
        never reads this."""
        one = jnp.ones((), dtype)
        return (one / jnp.array(2, dtype),)


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
    `zorch.sumcheck.prover.SumcheckRound`); emits a `RoundMsg`."""

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

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """Round polynomial over [0..degree], shape (degree+1, *batch):
        s[u] = sum_x' combine(f_u for each factor). One batched reduction."""
        return jnp.sum(self._combine(*factors_on_domain(state, self.degree)), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, RoundMsg]:
        msg = self._round_poly(state)
        transcript, r = transcript.observe_and_sample(msg, 1)
        state = fold(state, r[0])
        return state, transcript, RoundMsg(msg, r[0])


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
        # State order is LogupSumcheckRound's: [eq, n0, d1, n1, d0].
        state: list[Array] = [
            expand_eq_to_hypercube(eval_point, one),
            self.layer.numerator_0,
            self.layer.denominator_1,
            self.layer.numerator_1,
            self.layer.denominator_0,
        ]
        final_state, transcript, msgs = prove(
            LogupSumcheckRound(lam), state, transcript
        )
        round_polys, point = msgs.round_poly, msgs.challenge

        _, n0, d1, n1, d0 = (factor[0] for factor in final_state)
        transcript, r = transcript.observe_and_sample(jnp.stack([n0, n1, d0, d1]), 1)
        r = r[0]
        num_eval, den_eval, eval_point = fold_carry(n0, n1, d0, d1, point, r)

        proof = LayerProof(round_polys, point, n0, n1, d0, d1)
        return (num_eval, den_eval, eval_point), transcript, proof


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _summand: type[SumcheckSummand] = LogupSumcheckRound
    _gruen_summand: type[GruenSummand] = LogupSummand
    _sumcheck_round: type[ProverRound] = LogupSumcheckRound
    _layer_round: type[ProverRound] = GkrLayerRound
