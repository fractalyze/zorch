# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR prover.

`LogupSumcheckRound` is one per-variable sumcheck round whose summand is the
LogUp combine `eq * (lam*(n0*d1 + n1*d0) + d0*d1)` over five MLE factors
`[eq, n0, d1, n1, d0]` -- the sibling of the product `zorch.sumcheck.prover.
SumcheckRound`. Its message is a `RoundMsg(round_poly, challenge)`: the challenge
the round sampled rides in the message so the message-agnostic `fold_rounds`
driver collects the evaluation point alongside the round polynomials, in one
forward pass (the verifier re-derives the same point from the round polys).

`GkrLayerRound` is one GKR layer: it runs the layer's per-variable LogUp sumcheck
(via `fold_rounds` over `LogupSumcheckRound`), then reduces the numerator and
denominator claims across the child selector. The whole GKR prover is
`ProveChain([GkrLayerRound(l) for l in reversed(layers[:-1])])` -- the interaction
floor outward to the input, one bound variable per layer.

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

import jax
import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr.circuit import GkrLayer, LogUpGkrOutput
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.prover import factors_on_domain, fold
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2).
_DEGREE = 3
_NUM_FACTORS = 5  # [eq, n0, d1, n1, d0]


def logup_combine(
    lam: Array, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array
) -> Array:
    """The LogUp summand `eq * (lam*(n0*d1 + n1*d0) + d0*d1)`.

    Module-level so `LogupSumcheckRound` here and the GKR verifier's oracle
    evaluate the *same* expression -- a drift between them would break soundness
    silently.
    """
    return eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)


@dataclass(frozen=True)
class RoundMsg:
    """One per-variable round's output: the round polynomial plus the challenge
    it induced. The challenge is the prover's own Fiat-Shamir randomness (the
    verifier re-derives it from `round_poly`); bundling it lets `fold_rounds`
    collect the evaluation point without a separate channel or a transcript
    replay."""

    round_poly: Array
    challenge: Array


@partial(jax.tree_util.register_dataclass, data_fields=["lam"], meta_fields=[])
@dataclass(frozen=True)
class LogupSumcheckRound(Round):
    """Per-variable sumcheck round for the LogUp combine (sibling of the product
    `zorch.sumcheck.prover.SumcheckRound`); emits a `RoundMsg`."""

    # Batching challenge; fixed across a layer's variable-rounds.
    lam: Array

    def _combine(self, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array) -> Array:
        return logup_combine(self.lam, eq, n0, d1, n1, d0)

    def _round_poly(self, state: Sequence[Array]) -> Array:
        """Round polynomial over [0..degree], shape (degree+1, *batch):
        s[u] = sum_x' combine(f_u for each factor). One batched reduction."""
        if len(state) != _NUM_FACTORS:
            raise ValueError(
                f"state must hold {_NUM_FACTORS} factors [eq, n0, d1, n1, d0], "
                f"got {len(state)}"
            )
        return jnp.sum(self._combine(*factors_on_domain(state, _DEGREE)), axis=-1)

    def __call__(
        self, state: Sequence[Array], transcript: Transcript
    ) -> tuple[list[Array], Transcript, RoundMsg]:
        msg = self._round_poly(state)
        transcript = transcript.observe(msg)
        transcript, r = transcript.sample(1)
        state = fold(state, r[0])
        return state, transcript, RoundMsg(msg, r[0])


@dataclass(frozen=True)
class LayerProof:
    """One GKR layer's sumcheck transcript: round polynomials + final openings."""

    round_polys: Array  # (num_variables, degree + 1), each round's univariate
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
        final_state, transcript, msgs = fold_rounds(
            LogupSumcheckRound(lam), state, transcript, self.layer.num_variables
        )
        round_polys = jnp.stack([m.round_poly for m in msgs])
        point = jnp.stack([m.challenge for m in msgs])

        _, n0, d1, n1, d0 = (factor[0] for factor in final_state)
        transcript = transcript.observe(jnp.stack([n0, n1, d0, d1]))
        transcript, r = transcript.sample(1)
        r = r[0]
        num_eval = n0 + (n1 - n0) * r
        den_eval = d0 + (d1 - d0) * r
        # MSB-first point + the pyramid's child selector as the low (last) bit.
        eval_point = jnp.concatenate([point, jnp.atleast_1d(r)])

        proof = LayerProof(round_polys, n0, n1, d0, d1)
        return (num_eval, den_eval, eval_point), transcript, proof
