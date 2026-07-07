# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Evaluation domains for sumcheck round polynomials.

A round polynomial goes on the wire as ascending coefficients (the form
verifier.CoeffsSumcheckRound checks). EvalDomain names the points a prover samples
it at and owns the map from those samples to coefficients: a finite node set — the
Gruen set {0, 1, *extra, eq_root(z)}, or the naturals — optionally led by the value
at infinity (the leading coefficient, cheap for a product since it is the product of
the factor slopes). extend_to_round_domain lifts a linear pair onto the Û_d sample
domain; product_round_poly / product_round_coeffs build the baseline product round;
ProductRound / prove_product are the product sumcheck (Algorithm 1).
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _naturals(n: int, dtype: Any) -> Array:
    """The field-typed naturals {0, 1, …, n−1} (a stack, not iota — unsupported for
    extension dtypes)."""
    return jnp.stack([jnp.array(k, dtype) for k in range(n)])


def _finite_coeff_matrix(nodes: Array) -> Array:
    """Value → ascending-coefficient matrix (n, n) for a degree-(n−1) polynomial at
    the n finite nodes, bridged through the naturals then the inverse Vandermonde."""
    naturals, inv_vand = _interp_constants(nodes.shape[0] - 1, nodes.dtype)
    lagrange = jax.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, nodes)
    return jnp.dot(inv_vand, lagrange)


@dataclass(frozen=True)
class EvalDomain:
    """The sample points of a round polynomial and the map from those samples to
    ascending coefficients.

    nodes are the finite sample points — the Gruen set {0, 1, *extra, eq_root(z)},
    or (when None) the naturals {0..}. leading prepends the value at infinity (the
    leading coefficient) as the first sample. The polynomial's degree and field come
    from the samples themselves, so nothing but the node shape is fixed here."""

    nodes: Array | None = None
    leading: bool = False

    def coeff_matrix(self) -> Array:
        """Value → coefficient matrix for the explicit finite nodes — what a driver
        precomputes once per round. The leading / naturals domain has no fixed size,
        so it reads its degree off the values instead: use to_coeffs."""
        assert self.nodes is not None, "leading / naturals domain has no fixed matrix"
        return _finite_coeff_matrix(self.nodes)

    def to_coeffs(self, values: Array) -> Array:
        """Ascending coefficients from this domain's samples of a round polynomial;
        the degree (len−1) and field come from values."""
        if not self.leading:
            return jnp.dot(self.coeff_matrix(), values)
        # [∞, *finite]: the ∞ sample is the leading coefficient c_d, and the finite
        # samples (naturals {0..d−1} unless given) interpolate the residual p − c_d·xᵈ.
        v_inf, finite = values[0], values[1:]
        d = finite.shape[0]
        nodes = self.nodes if self.nodes is not None else _naturals(d, values.dtype)
        low = jnp.dot(_finite_coeff_matrix(nodes), finite - v_inf * nodes**d)
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
    evals = jnp.sum(jnp.prod(lifted, axis=0), axis=1)
    return EvalDomain(leading=True).to_coeffs(evals)


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
