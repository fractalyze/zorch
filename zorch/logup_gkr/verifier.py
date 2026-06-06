# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense LogUp-GKR verifier -- the dual of the prover chain.

A `GkrLayerRound` replays one layer's per-variable sumcheck (the agnostic
`zorch.verify` driver over `zorch.sumcheck.verifier.SumcheckRound`), checks the
LogUp oracle at the bound point via the shared `logup_combine`, then reduces the
claim across the child selector. The whole GKR verifier is
`VerifyChain([GkrLayerRound() for _ in layer_proofs])`, threading the same
`(num_eval, den_eval, eval_point)` carry the prover does and ANDing every layer's
check. The eq factor of the oracle is evaluated with the O(n) `eval_eq`, so
verification stays succinct (no 2**n weight vector).

It stops at the reduced point-claim. The final `claim == leaf_mle(point)` check
needs a PCS opening of the input trace and is the consumer's, keeping this block
PCS-agnostic; the roundtrip tests close it directly against the dense leaf MLE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr.prover import Carry, LayerProof, logup_combine
from zorch.poly.eq import eval_eq
from zorch.round import Round
from zorch.sumcheck.verifier import SumcheckRound as SumcheckVerifierRound
from zorch.transcript import Transcript
from zorch.verify import verify

if TYPE_CHECKING:
    from zorch.round import ChainVerifierRound

_DEGREE = 3  # LogUp combine round-polynomial degree (eq * deg-2 bracket).


class GkrLayerRound(Round):
    """Verify one GKR layer; the chain of these is the GKR verifier."""

    def __call__(
        self, carry: Carry, layer_proof: LayerProof, transcript: Transcript
    ) -> tuple[Carry, Transcript, Array]:
        num_eval, den_eval, eval_point = carry
        n0, n1 = layer_proof.numerator_0, layer_proof.numerator_1
        d0, d1 = layer_proof.denominator_0, layer_proof.denominator_1
        transcript, lam = transcript.sample(1)
        lam = lam[0]
        claim = lam * num_eval + den_eval
        point, final_claim, transcript, ok_sc = verify(
            SumcheckVerifierRound(_DEGREE), claim, layer_proof.round_polys, transcript
        )
        # LogUp oracle: the reduced claim equals the combine at the bound point.
        # eq is MSB-first on both sides, so the points align with no flip.
        eq_eval = eval_eq(eval_point, point)
        combined = logup_combine(lam, eq_eval, n0, d1, n1, d0)
        ok = ok_sc & (combined == final_claim)

        transcript, r = transcript.observe_and_sample(jnp.stack([n0, n1, d0, d1]), 1)
        r = r[0]
        num_eval = n0 + (n1 - n0) * r
        den_eval = d0 + (d1 - d0) * r
        eval_point = jnp.concatenate([point, jnp.atleast_1d(r)])
        return (num_eval, den_eval, eval_point), transcript, ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ChainVerifierRound] = GkrLayerRound
