# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Evaluation domain for sumcheck round polynomials.

A round polynomial (degree ≤ d) goes on the wire as evaluations over some domain,
with one node omitted and recovered from s(0)+s(1)=claim. This module owns that
domain machinery. It currently carries U_d = {∞, 0, 1, …, d−1}, the domain the
eq-poly family uses: ∞ is the leading coefficient (the product of the factor
slopes, cheaper than a large finite node), and provers drop the u=1 node (Û_d).
extend_to_round_domain lifts a linear pair onto it; product_round_poly builds a
product round message; ProductRound / prove_product are the baseline product
sumcheck (Algorithm 1) that emits them.

The natural {0..d} and coefficient forms still live in prover.py / verifier.py;
folding every form behind one domain parameter is the extension point the verifier
duals will motivate (they need the same recover-omitted-node + eval-at-challenge).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
)
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


@cache
def _interp_constants(degree: int, dtype: Any) -> tuple[Array, Array]:
    """Lagrange naturals ({0..degree}) and the inverse Vandermonde, memoized per
    (degree, dtype): compute_inv_vandermonde is an O(degree²) host build, pure
    redundant work per round without the memo."""
    # Force concrete eval: @cache would otherwise cache a tracer built inside a
    # jit trace, which then escapes it (UnexpectedTracerError). The constants are
    # trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(degree + 1)])
        inv_vand = compute_inv_vandermonde(degree, dtype)
    return naturals, inv_vand


def coeffs_from_nodes(nodes: Array, degree: int) -> Array:
    """Value → ascending-coefficient matrix (degree+1, degree+1) for a round poly at
    degree+1 finite nodes — bridged through the naturals {0..degree}, so any
    node set reaches the form verifier.CoeffsSumcheckRound checks."""
    naturals, inv_vand = _interp_constants(degree, nodes.dtype)
    lagrange = jax.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, nodes)
    return jnp.dot(inv_vand, lagrange)


def coeffs_from_round_domain(evals: Array, degree: int) -> Array:
    """Ascending coefficients of a degree-degree polynomial given as its values
    on the round domain [∞, 0, 1, …, degree−1] (the ∞ entry is the leading
    coefficient). Splits p = q + c_deg·xᵈ: the finite residuals p(j) − c_deg·jᵈ
    interpolate the degree−1 part q, and c_deg is the ∞ value."""
    v_inf, v_finite = evals[0], evals[1:]
    nodes = jnp.stack([jnp.array(k, evals.dtype) for k in range(degree)])
    node_pow = jnp.stack([jnp.array(k, evals.dtype) ** degree for k in range(degree)])
    low = jnp.dot(coeffs_from_nodes(nodes, degree - 1), v_finite - v_inf * node_pow)
    return jnp.concatenate([low, jnp.atleast_1d(v_inf)])


def extend_to_round_domain(
    p0: Array, p1: Array, d: int, *, skip_one: bool = False
) -> Array:
    """Lift a linear pair (p(0), p(1)) onto U_d = [∞, 0, 1, …, d−1] (or Û_d, u=1
    dropped, when skip_one). p(∞) is the slope p(1)−p(0);
    p(u) = p(0) + u·(p(1)−p(0)). The leading axis indexes the domain."""
    diff = p1 - p0
    base = jnp.stack([diff, p0]) if skip_one else jnp.stack([diff, p0, p1])
    if d <= 2:
        return base
    # Python-int multiplier avoids a field-dtype iota (unsupported in the fork).
    rest = jnp.stack([p0 + diff * u for u in range(2, d)], axis=0)
    return jnp.concatenate([base, rest], axis=0)


def product_round_poly(stacked: Array) -> Array:
    """Round message s = Σₓ Πₖ fₖ over Û_m for the m stacked multilinears, shape
    (m,)."""
    m = stacked.shape[0]
    pairs = jnp.reshape(stacked, (m, 2, -1))
    p0, p1 = pairs[:, 0, :], pairs[:, 1, :]
    lifted = jax.vmap(lambda a, b: extend_to_round_domain(a, b, m, skip_one=True))(
        p0, p1
    )
    return jnp.sum(jnp.prod(lifted, axis=0), axis=1)


def product_round_coeffs(stacked: Array) -> Array:
    """Ascending coefficients of the degree-m product round polynomial for m stacked
    factors: the same Σ_x' Πₖ fₖ as product_round_poly but evaluated over the full
    round domain [∞, 0, 1, …, m−1] so it is fully determined, then mapped to
    coefficients — the wire form verifier.CoeffsSumcheckRound checks."""
    m = stacked.shape[0]
    pairs = jnp.reshape(stacked, (m, 2, -1))
    p0, p1 = pairs[:, 0, :], pairs[:, 1, :]
    lifted = jax.vmap(lambda a, b: extend_to_round_domain(a, b, m))(p0, p1)
    return coeffs_from_round_domain(jnp.sum(jnp.prod(lifted, axis=0), axis=1), m)


def fold_stacked(stacked: Array, r: Array) -> Array:
    """Standard sumcheck fold of the leading variable on a stacked (m, N) array,
    halving the width — shared by the product round and the small-value transition."""
    pairs = jnp.reshape(stacked, (stacked.shape[0], 2, -1))
    p0 = pairs[:, 0, :]
    return (pairs[:, 1, :] - p0) * r + p0


class ProductRound(Round):
    """Product sumcheck round: send Πₖ fₖ over Û_d, fold every factor at the
    challenge. product_round_coeffs gives the same round in the coefficient wire
    form for verifier.CoeffsSumcheckRound."""

    def __call__(
        self, stacked: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array]:
        msg = product_round_poly(stacked)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return fold_stacked(stacked, r[0]), transcript, msg


def prove_product(
    p_initial: Array, transcript: Transcript
) -> tuple[Array, Transcript, list[Array]]:
    """Linear-time product sumcheck over all l variables, Û_d messages."""
    rounds = log2_strict_usize(p_initial.shape[1])
    return fold_rounds(ProductRound(), p_initial, transcript, rounds)
