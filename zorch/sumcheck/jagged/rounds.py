# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic per-round helpers for jagged sumcheck: the virtual-mass
correction, the Gruen coefficient-form round polynomial, and the eq-slice
expansion. Degree-agnostic — they take the interpolation constants
(`naturals`/`inv_vand`) as arguments (also the round marker's operands, so the
crossing stays inline here rather than routing through `gruen.round_coeffs`,
which would rebuild them and DCE the operands out of the composite). The LSB
bind is `sumcheck.domain.fold(..., msb=False)`. The LogUp-specific combine +
plane bodies live in `zorch.logup_gkr._jagged_rounds`."""

from __future__ import annotations

import frx
import frx.numpy as jnp
from frx import Array

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_lagrange_basis


def _virtual_mass_correction(pad_adj: Array, eq_sum: Array) -> Array:
    """The virtual (non-materialized) mass a round adds back in closed form.

    The sumcheck runs only over the materialized rows; every non-materialized
    position holds the fold-neutral fraction (n=0, d=1), whose LogUp summand
    collapses to just its eq weight. The eq weights of the full remaining
    hypercube sum to `pad_adj`, so the virtual positions contribute exactly
    `pad_adj - eq_sum` (the full mass minus the materialized `eq_sum`) -- added
    back by `_round_coeffs` in closed form rather than iterated over."""
    return pad_adj - eq_sum


def _round_coeffs(
    eval_zero: Array,
    eval_half: Array,
    eq_sum: Array,
    eq_adj: Array,
    pad_adj: Array,
    z_cur: Array,
    claim: Array,
    naturals: Array,
    inv_vand: Array,
) -> Array:
    """Round polynomial coefficients from the materialized {0, 1/2} sums.

    Every non-materialized position contributes exactly its eq weight, and
    the eq weights of the full remaining hypercube sum to `pad_adj`, so the
    virtual mass is `pad_adj - eq_sum`: at u=0 it carries the current
    variable's eq factor `(1 - z_cur)`; at the doubled u=1/2 scale each
    virtual pair contributes `den_h = 4` at weight `eq_h = eq_rest`, and the
    doubled products overcount s(1/2) by 8. `eq_adj` is the row-eq residual
    scalar once the row variables are exhausted (1 before that).

    The interpolant through {0, 1, 1/2, b} crosses to coefficients via the
    natural domain: Lagrange-evaluate it at `naturals` ({0..degree}), then
    `inv_vand` maps values to coefficients. Both are hoisted by the caller
    -- they only depend on the degree.
    """
    dtype = claim.dtype
    one = jnp.ones((), dtype)
    correction = _virtual_mass_correction(pad_adj, eq_sum)
    s_zero = (eval_zero + correction * (one - z_cur)) * eq_adj
    s_half = (
        (eval_half + correction * jnp.array(4, dtype)) / jnp.array(8, dtype) * eq_adj
    )
    s_one = claim - s_zero
    b_root = (one - z_cur) / (one - jnp.array(2, dtype) * z_cur)
    xs = jnp.stack([jnp.zeros((), dtype), one, one / jnp.array(2, dtype), b_root])
    ys = jnp.stack([s_zero, s_one, s_half, jnp.zeros((), dtype)])
    lagrange = frx.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, xs)
    return jnp.dot(inv_vand, jnp.dot(lagrange, ys))


def _expand_eq_slice(eval_point: Array, niv: int, *, row: bool) -> Array:
    """`expand_eq_to_hypercube` over the row (`eval_point[niv:]`) or batch
    (`eval_point[:niv]`) coordinate block, traced into the whole-layer jit. `niv`
    (and hence the slice bounds + output length) rides static."""
    coords = eval_point[niv:] if row else eval_point[:niv]
    return expand_eq_to_hypercube(coords, jnp.ones((), eval_point.dtype))
