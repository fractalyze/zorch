# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Evaluation domains for sumcheck round polynomials.

A round polynomial goes on the wire as ascending coefficients (the form
verifier.CoeffsSumcheckRound checks). EvalDomain names the points a prover samples
it at and owns the map from those samples to coefficients: a finite node set — the
Gruen set {0, 1, *extra, eq_root(z)}, or the naturals — optionally led by the value
at infinity (the leading coefficient, cheap for a product since it is the product of
the factor slopes). extend_to_round_domain lifts a linear pair onto the Û_d sample
domain; product_round_poly / product_round_coeffs build a product round message over
it (the prover Rounds that emit them live in sqrt_space).
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, reduce
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array

from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
)


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
    """The field-typed naturals {0, 1, …, n−1}. Built in the base field — an iota
    over an extension dtype is unsupported in the fork, and integer nodes live in
    the prime field anyway; extension callers promote at multiply time. Same pattern
    as compute_inv_vandermonde."""
    try:
        base = zk_dtypes.efinfo(dtype).base_field_dtype
    except ValueError:
        base = dtype
    return jnp.array(list(range(n)), base)


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
        if self.nodes is None:
            # Naturals domain: _finite_coeff_matrix(_naturals(d)) is exactly the
            # inverse Vandermonde — its Lagrange build over the naturals collapses to
            # the identity — so skip that identity vmap + matmul and use inv_vand.
            nodes, cmat = _interp_constants(d - 1, values.dtype)
        else:
            nodes, cmat = self.nodes, _finite_coeff_matrix(self.nodes)
        low = jnp.dot(cmat, finite - v_inf * nodes**d)
        return jnp.concatenate([low, jnp.atleast_1d(v_inf)])

    def sample(self, p0: Array, p1: Array) -> Array:
        """Sample the linear factor p(x) = p0 + x·(p1−p0) at this domain's points
        [∞ if leading, *nodes]: p(∞) = slope p1−p0, p(node) = p0 + node·slope. The
        leading axis indexes the domain. Requires explicit nodes — a sampling domain
        must be concrete (the naturals-sized domain is an output-only coeff map)."""
        assert self.nodes is not None, "a sampling domain needs an explicit node set"
        diff = p1 - p0
        finite = p0[None] + self.nodes.reshape((-1,) + (1,) * p0.ndim) * diff[None]
        return jnp.concatenate([diff[None], finite]) if self.leading else finite


def extend_to_round_domain(
    p0: Array, p1: Array, d: int, *, skip_one: bool = False
) -> Array:
    """Lift a linear pair (p(0), p(1)) onto U_d = [∞, 0, 1, …, d−1] (or Û_d, u=1
    dropped, when skip_one). p(∞) is the slope p(1)−p(0);
    p(u) = p(0) + u·(p(1)−p(0)). The leading axis indexes the domain."""
    diff = p1 - p0
    if d == 1:  # U_1 = [∞, 0]: u=1 is out of range {0..d−1}, so p1 is not a node
        return jnp.stack([diff, p0])
    base = jnp.stack([diff, p0]) if skip_one else jnp.stack([diff, p0, p1])
    if d == 2:
        return base
    # Python-int multiplier avoids a field-dtype iota (unsupported in the fork).
    rest = jnp.stack([p0 + diff * u for u in range(2, d)], axis=0)
    return jnp.concatenate([base, rest], axis=0)


def _product(*factors: Array) -> Array:
    """The product summand Πₖ fₖ — the default combine for summand_evals."""
    return reduce(operator.mul, factors)


def natural_domain(degree: int, dtype: Any) -> EvalDomain:
    """The natural evaluation domain {0, 1, …, degree}: the round poly sent as its
    plain values [s(0), …, s(degree)] — the wire form verifier.SumcheckRound checks
    and the default domain of the generic StandardRound. Nodes live in the base
    field (an integer node is a base-field element; extension factors promote at
    multiply), reproducing the list prover's per-point lift byte-for-byte."""
    return EvalDomain(_naturals(degree + 1, dtype))


def uhat_domain(degree: int, dtype: Any) -> EvalDomain:
    """The compressed product round domain Û_degree = {∞, 0, 2, …, degree−1}:
    ∞-leading, u=1 dropped (the verifier recovers s(1) from s(0)+s(1)=claim). The
    default sampling domain for the eq-poly / sqrt-space engines."""
    nat = _naturals(degree, dtype)
    return EvalDomain(jnp.concatenate([nat[:1], nat[2:]]), leading=True)


def summand_evals(
    stacked: Array, combine: Callable[..., Array], domain: EvalDomain
) -> Array:
    """Σ_x' combine(f₁, …, f_m)(x') per point of `domain`: the m stacked factors
    sampled at the domain's points (domain.sample), combined, then summed. The one
    reduction body the round-message builders share — generic over BOTH the summand's
    `combine` (Πₖ, LogUp, …) and the evaluation domain (∞-leading Û, Gruen, {0,½},
    {0,2,4}, …).

    A leading (∞) node encodes s(∞) = combine(*slopes), the true leading coefficient
    only for a HOMOGENEOUS combine — every monomial a product of exactly `degree`
    factors (a plain product, or the LogUp combine); a mixed-degree combine like
    E·(A·B − C) is not. A finite domain carries no such restriction — any summand
    samples cleanly on it."""
    m = stacked.shape[0]
    pairs = jnp.reshape(stacked, (m, 2, -1))
    p0, p1 = pairs[:, 0, :], pairs[:, 1, :]
    lifted = jax.vmap(domain.sample)(p0, p1)
    return jnp.sum(combine(*lifted), axis=1)


def product_round_poly(stacked: Array) -> Array:
    """Round message s = Σₓ Πₖ fₖ over Û_m for the m stacked multilinears, shape
    (m,) — the product summand on the compressed domain."""
    return summand_evals(
        stacked, _product, uhat_domain(stacked.shape[0], stacked.dtype)
    )


def product_round_coeffs(stacked: Array) -> Array:
    """Ascending coefficients of the degree-m product round polynomial for m stacked
    factors: the same Σ_x' Πₖ fₖ as product_round_poly but sampled over the full round
    domain [∞, 0, 1, …, m−1] so it is fully determined, then mapped to coefficients —
    the wire form verifier.CoeffsSumcheckRound checks."""
    m = stacked.shape[0]
    full = EvalDomain(_naturals(m, stacked.dtype), leading=True)  # [∞, 0, 1, …, m−1]
    return EvalDomain(leading=True).to_coeffs(summand_evals(stacked, _product, full))


def fold_stacked(stacked: Array, r: Array) -> Array:
    """Standard sumcheck fold of the leading variable on a stacked (m, N) array,
    halving the width — shared by the product round and the small-value transition."""
    pairs = jnp.reshape(stacked, (stacked.shape[0], 2, -1))
    p0 = pairs[:, 0, :]
    return (pairs[:, 1, :] - p0) * r + p0
