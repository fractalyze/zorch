# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Gruen eq-factor round-poly compression: the coefficient-form assembly.

A sumcheck whose summand carries the current variable's eq factor
``eq_factor(t, z) = t·z + (1-t)(1-z)`` gets two round-poly evaluations for
free (Gruen, https://eprint.iacr.org/2024/108): the claim identity
``s(1) = claim - s(0)``, and a zero at the factor's root
``b = (1-z)/(1-2z)`` (`zorch.poly.eq.eq_root`), which both sides derive from
``z`` alone. A degree-d round therefore materializes only d-1 evaluations --
``s(0)`` plus d-2 points of the engine's choosing -- where value form on the
natural domain would materialize d+1.

`round_coeffs` is that assembly, parameterized by the engine's extra points:
the interpolant through ``{0, 1, *extra, b}`` crosses to the coefficient-form
wire encoding via the natural domain -- Lagrange-evaluate it on ``{0..d}``,
then the inverse Vandermonde maps natural values to ascending coefficients
(the form `zorch.sumcheck.verifier.CoeffsSumcheckRound` checks, node-set-
agnostic). The known instances:

- LogUp-GKR jagged (`zorch.logup_gkr.jagged_prover`): extra ``{1/2}``,
  degree 3.
- the jagged zerocheck engines in consumers: extra ``{2, 4}``, degree 4.

The next round's claim is the coefficients evaluated at the sampled challenge
(`zorch.poly.univariate.eval_coeffs`), and the round's bound eq mass
accumulates by ``eq_factor(r, z)`` -- both already shared definitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import eq_factor, eq_root
from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
)


class GruenSummand(Protocol):
    """The seam a Gruen-compressed jagged round needs from its summand: the
    round-poly ``degree`` and the ``degree - 2`` extra evaluation points the
    engine materializes beyond s(0) — t = 1 and the eq-factor root come free
    (the claim identity and the Gruen zero), so a degree-d round materializes
    exactly d - 1 evaluations. The round LOOP stays the consumer's (host loop
    vs fixed-shape scan — the engines' module docstrings own that choice);
    this protocol pins the vocabulary the shared assembly below reads. The
    known instances: `zorch.logup_gkr.prover.LogupSummand` (degree 3, extra
    ``{1/2}``) and the jagged zerocheck engines in consumers (degree 4,
    extra ``{2, 4}``).

    Every jagged summand also carries a **padding correction** — the
    sumcheck runs over materialized positions only, and the non-materialized
    ones contribute in closed form: LogUp's fold-neutral fraction (n=0, d=1)
    collapses to its eq weight
    (`zorch.logup_gkr.jagged_prover._virtual_mass_correction`), the
    zerocheck zero-extension row to its constant ``C(0_row)`` removed via
    the virtual geq. The correction's math is each summand's own; the
    concept is this seam's."""

    @property
    def degree(self) -> int: ...

    def extra_ts(self, dtype: Any) -> tuple[Array, ...]: ...


@cache
def _interp_constants(degree: int, dtype: Any) -> tuple[Array, Array]:
    """Lagrange ``naturals`` ({0..degree}) and the inverse Vandermonde,
    memoized per (degree, dtype): `compute_inv_vandermonde` is an O(degree^2)
    numpy coefficient build, pure redundant host work per round without the
    memo."""
    # Force concrete eval: `@cache` memoizes the result, so building it inside
    # a jit trace would cache a tracer that then escapes the trace
    # (UnexpectedTracerError). The constants are trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(degree + 1)])
        inv_vand = compute_inv_vandermonde(degree, dtype)
    return naturals, inv_vand


def interp_matrix(extra_ts: Sequence[Array], z: Array) -> Array:
    """The Gruen value-to-coefficient matrix for one round: maps the value
    vector ``[s(0), s(1), *extra_ys, 0]`` on the domain ``{0, 1, *extra_ts,
    eq_root(z)}`` to ascending coefficients, shape ``(degree + 1, degree + 1)``
    with ``degree = len(extra_ts) + 2``.

    Depends only on ``z`` and the static ``extra_ts``, so a fixed-shape scan
    driver precomputes it per round OUTSIDE the scan (``jax.vmap`` over the
    stacked round coordinates) and feeds it through the scan's xs; a
    host-relaunch driver just calls `round_coeffs`, which composes this with
    the value assembly."""
    dtype = z.dtype
    one = jnp.ones((), dtype)
    zero = jnp.zeros((), dtype)
    xs = jnp.stack([zero, one, *extra_ts, eq_root(z)])
    naturals, inv_vand = _interp_constants(len(extra_ts) + 2, dtype)
    lagrange = jax.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, xs)
    return jnp.dot(inv_vand, lagrange)


def round_coeffs_from_matrix(
    matrix: Array,
    s_zero: Array,
    claim: Array,
    extra_ys: Sequence[Array],
) -> Array:
    """Assemble one round polynomial from a prebuilt `interp_matrix`.

    ``s_zero`` is the engine's (already corrected) s(0), ``extra_ys`` its
    evaluations at the matrix's ``extra_ts`` points, and ``claim`` the running
    claim -- the zero at the eq-factor root is implicit in the matrix's last
    column. The values may carry leading batch axes (e.g. one evaluation set
    per chip); they broadcast into the value vector, and the result is
    ``(degree + 1, *batch)``."""
    if matrix.shape[-1] != len(extra_ys) + 3:
        raise ValueError(
            f"matrix expects {matrix.shape[-1] - 3} extra evaluations "
            f"(domain {{0, 1, ..., b}}), got {len(extra_ys)}"
        )
    zero = jnp.zeros((), claim.dtype)
    ys = jnp.stack(jnp.broadcast_arrays(s_zero, claim - s_zero, *extra_ys, zero))
    return jnp.dot(matrix, ys)


def round_coeffs(
    s_zero: Array,
    claim: Array,
    extra_ts: Sequence[Array],
    extra_ys: Sequence[Array],
    z: Array,
) -> Array:
    """Assemble one Gruen-compressed round polynomial in coefficient form.

    ``s_zero`` is the engine's (already corrected) s(0), ``extra_ys`` its
    evaluations at the ``extra_ts`` points, ``claim`` the running claim, and
    ``z`` the current variable's point coordinate. The interpolation domain is
    ``{0, 1, *extra_ts, eq_root(z)}`` with values ``{s_zero, claim - s_zero,
    *extra_ys, 0}`` -- degree ``len(extra_ts) + 2``. Returns ascending
    coefficients, shape ``(degree + 1,)``.

    The one-call composition of `interp_matrix` + `round_coeffs_from_matrix`
    for a driver with ``z`` in hand per round. Un-jitted on purpose: it traces
    into whichever kernel owns the round (the consumers' whole-layer / scan
    zones). The interpolation constants resolve concretely inside any
    enclosing trace (`_interp_constants`), so they bake in as closure
    constants, never operands."""
    if len(extra_ts) != len(extra_ys):
        raise ValueError(
            f"extra_ts and extra_ys must pair up: got {len(extra_ts)} points "
            f"and {len(extra_ys)} evaluations"
        )
    return round_coeffs_from_matrix(interp_matrix(extra_ts, z), s_zero, claim, extra_ys)


def fold_round_scalars(
    poly: Array, r: Array, mass: Array, z: Array
) -> tuple[Array, Array]:
    """The post-round scalar fold every Gruen-compressed round ends with: the
    next claim is the coefficient-form round polynomial evaluated at the
    sampled challenge, and the running bound-eq mass gains the bound
    variable's eq factor. Returns ``(claim', mass')``.

    One definition so a prover's round loop and its unrolled oracle cannot
    drift out of byte-equality — every engine driving this assembly (the
    LogUp jagged prover, `zorch.logup_gkr.jagged_prover`; consumers' jagged
    zerocheck engines) ends its round with exactly this pair."""
    return eval_coeffs(poly, r), mass * eq_factor(r, z)
