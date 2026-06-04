# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Stacked dense-open: adapt a BaseFold matrix opening to a full MLE evaluation.

The dense MLE `D` is committed as an `[S, K]` matrix (S=2^log_s rows,
K=2^log_k columns, log_m=log_s+log_k).  A caller supplies `z_final ∈ F^{log_m}`
and wants `D(z_final)`.

Split `z_final → (z_K [leading log_k vars], stack_point [trailing log_s vars])`.
BaseFold opens `D` at `stack_point`, returning per-column evals `values[k] =
D_k(stack_point)`.  We then eq-combine across columns:

    D(z_final) = Σ_k  eq(z_K, k)  ·  values[k]

using a trace-time-unrolled `functools.reduce` (never `jnp.sum`/`jnp.prod` over
EF — those abort the ZKX interpreter; see docs/poly.md).
"""
from __future__ import annotations

import functools
import operator

import jax.numpy as jnp
from jax import Array

from zorch.pcs.basefold.config import BasefoldProof
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.pcs.jagged.dense import JaggedLayout
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript


def _split(z_final: Array, layout: JaggedLayout) -> tuple[Array, Array]:
    """Split `z_final` into (z_K, stack_point).

    `z_K` addresses the K columns (leading `log_k` variables, MSB);
    `stack_point` addresses the S rows within a column (trailing `log_s` vars).
    This matches the dense layout: `d_full = mle.T.reshape(-1)` stores index
    `k*S + s`, so the K-bits are the MSBs of `z_final`.
    """
    log_k = layout.log_m - layout.log_s
    z_k = z_final[:log_k]
    stack_point = z_final[log_k:]
    return z_k, stack_point


def _combine(values: Array, z_k: Array, layout: JaggedLayout) -> Array:
    """Eq-combine K per-column evals into a single dense eval.

    weights[k] = eq(z_K, k) (the hypercube expansion at z_K).
    dense_eval = Σ_k weights[k] * values[k].

    The sum is trace-time unrolled (static layout.K) via functools.reduce —
    never jnp.sum over EF (ZKX abort).
    """
    dtype = z_k.dtype
    weights = expand_eq_to_hypercube(z_k, jnp.array(1, dtype))  # (K,)
    K = layout.K
    terms = [weights[k] * values[k] for k in range(K)]
    return functools.reduce(operator.add, terms)


def stacked_open(
    bf_prover: BasefoldProver,
    prover_data: BasefoldProverData,
    z_final: Array,
    layout: JaggedLayout,
    transcript: Transcript,
) -> tuple[Array, Array, BasefoldProof, Transcript]:
    """Open the stacked MLE at `z_final` -> `(dense_eval, values, proof, transcript)`.

    `values` is the `(K,)` array of per-column evals at `stack_point`.
    `dense_eval` is the scalar MLE evaluation at `z_final`.
    """
    z_k, stack_point = _split(z_final, layout)
    values, proof, transcript = bf_prover.open(prover_data, [stack_point], transcript)
    dense_eval = _combine(values, z_k, layout)
    return dense_eval, values, proof, transcript


def stacked_verify(
    bf_verifier: BasefoldVerifier,
    commitment: Array,
    z_final: Array,
    dense_eval: Array,
    values: Array,
    proof: BasefoldProof,
    layout: JaggedLayout,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    """Verify a stacked opening.  Returns `(ok, transcript)`.

    Checks:
    1. BaseFold verifies the K column evals at `stack_point`.
    2. The eq-combine of `values` by `z_K` matches `dense_eval`.
    """
    z_k, stack_point = _split(z_final, layout)
    ok_bf, transcript = bf_verifier.verify(
        commitment, [stack_point], values, proof, transcript
    )
    recomputed = _combine(values, z_k, layout)
    ok_combine = jnp.bool_(recomputed == dense_eval)
    return ok_bf & ok_combine, transcript
