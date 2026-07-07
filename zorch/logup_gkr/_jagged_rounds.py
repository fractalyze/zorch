# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-specific eager bodies of one jagged sumcheck round: the LogUp
combine (`_paired_sums`), the plane binds, and the row/boundary/interaction
fold+sum variants the composites and the reference oracle share. The
scheme-agnostic round helpers live in `zorch.sumcheck.jagged.rounds`."""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr._jagged_types import _DEGREE, _Planes, _RoundScalars
from zorch.logup_gkr.circuit import _pad_neutral
from zorch.logup_gkr.prover import LogupSummand
from zorch.poly.univariate import compute_inv_vandermonde
from zorch.sumcheck.jagged.rounds import _bind_lsb, _round_coeffs
from zorch.sumcheck.jagged.types import _InterpConsts


def _paired_sums(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_0: Array,
    eq_1: Array,
    lam: Array,
    live_pairs: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Materialized `(s(0), 8*s(1/2), eq mass)` over the stride-2 pairs.

    s(0) reads the even elements at their eq weight; the u=1/2 sum works on
    doubled values (`e0 + e1 = 2*e(1/2)` per factor, likewise eq), which
    `_round_coeffs` rescales. Both go through the shared `LogupSummand` combine
    so the summand cannot drift from the verifier oracle's.

    `live_pairs` masks the sums to the first `live_pairs` pairs -- the
    fixed-width round buffers (xla#179) carry a dead tail past the live state,
    and field addition is exact, so the masked full-width sum is bit-identical
    to the exact-width sum. None sums every pair (the exact-width layout).
    """
    summand = LogupSummand(lam)
    scalars = summand.combine_scalars()
    terms_zero = summand.combine(scalars, eq_0, n0[0::2], d1[0::2], n1[0::2], d0[0::2])
    eq_h = eq_0 + eq_1
    terms_half = summand.combine(
        scalars,
        eq_h,
        n0[0::2] + n0[1::2],
        d1[0::2] + d1[1::2],
        n1[0::2] + n1[1::2],
        d0[0::2] + d0[1::2],
    )
    if live_pairs is not None:
        mask = jnp.arange(terms_zero.shape[0]) < live_pairs
        terms_zero = jnp.where(mask, terms_zero, jnp.zeros_like(terms_zero))
        terms_half = jnp.where(mask, terms_half, jnp.zeros_like(terms_half))
        eq_h = jnp.where(mask, eq_h, jnp.zeros_like(eq_h))
    return jnp.sum(terms_zero), jnp.sum(terms_half), jnp.sum(eq_h)


def _bind_planes(planes: _Planes, alpha: Array) -> _Planes:
    return _Planes(
        *(_bind_lsb(a, alpha) for a in (planes.n0, planes.n1, planes.d0, planes.d1))
    )


def _round_poly_int(
    planes: _Planes,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_pairs: Array | None = None,
) -> Array:
    """The `sum_as_poly` step for the dense batch phase: the round
    univariate from the current state, no fold (the entry kernel of a
    round loop, before any challenge is bound).

    Fiat-Shamir-less by construction — the FS hop lives in `_fs_reduce`, appended
    after the round compute — so the body is pure field arithmetic. `eq_int` is
    sliced stride-2 once inside `_paired_sums`, over an even state. `live_pairs`
    masks the sum to the live pairs of a fixed-width state (see `_paired_sums`)."""
    eval_zero, eval_half, eq_sum = _paired_sums(
        planes.n0,
        planes.n1,
        planes.d0,
        planes.d1,
        eq_int[0::2],
        eq_int[1::2],
        scalars.lam,
        live_pairs=live_pairs,
    )
    return _round_coeffs(
        eval_zero,
        eval_half,
        eq_sum,
        scalars.eq_adj,
        scalars.pad_adj,
        scalars.z_cur,
        scalars.claim,
        consts.naturals,
        consts.inv_vand,
    )


def _fix_and_sum_int(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The `fix_and_sum` step for the dense batch phase: bind the previous
    round's challenge `alpha` (state size `m -> m/2`) **then** compute the next
    round's univariate at the halved size. Returns `(poly, planes, eq_int)` so the
    loop threads the folded state into the next round. The fold and the inner
    `_paired_sums` slice stride-2 twice."""
    planes = _bind_planes(planes, alpha)
    eq_int = _bind_lsb(eq_int, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _round_poly_row(
    planes: _Planes,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_pairs: Array | None = None,
) -> tuple[Array, _Planes]:
    """The row-variable round body shared by the row kernels: re-pad the four
    MLEs to the round's even layout (`gather`), look the per-pair batch eq
    weight up via `eq_int[col_index]`, and form the round univariate over the
    segment-local `eq_row` pairs. Returns `(poly, planes)` — the padded state the
    caller binds next round.

    The schedule (`gather`, `col_index`, `pair_index`) is host-built; the post-pad
    state is `gather`'s length (even), so the `_paired_sums` stride-2 stays valid.
    `live_pairs` masks the sum to the live pairs of a fixed-width schedule (see
    `_paired_sums`)."""
    n0, n1, d0, d1 = _pad_neutral(planes.n0, planes.n1, planes.d0, planes.d1, gather)
    w = eq_int[col_index]
    eval_zero, eval_half, eq_sum = _paired_sums(
        n0,
        n1,
        d0,
        d1,
        eq_row[pair_index * 2] * w,
        eq_row[pair_index * 2 + 1] * w,
        scalars.lam,
        live_pairs=live_pairs,
    )
    poly = _round_coeffs(
        eval_zero,
        eval_half,
        eq_sum,
        scalars.eq_adj,
        scalars.pad_adj,
        scalars.z_cur,
        scalars.claim,
        consts.naturals,
        consts.inv_vand,
    )
    return poly, _Planes(n0, n1, d0, d1)


def _fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The `fix_and_sum` step for the row-variable phase: bind the previous
    round's challenge `alpha` (state `2p_prev -> p_prev`, and `eq_row` halved in
    step) **then** re-pad to this round's layout and compute the next univariate.
    Returns `(poly, planes, eq_row)`.

    `eq_row` folds inside the kernel because the loop binds it at each round's
    end. The input state and `eq_row` enter even, and the `_pad_neutral` output is
    even, so all halvings stay valid."""
    planes = _bind_planes(planes, alpha)
    eq_row = _bind_lsb(eq_row, alpha)
    poly, planes = _round_poly_row(
        planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
    )
    return poly, planes, eq_row


def _fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """The row->batch handoff in one launch: bind the last row variable's
    challenge `alpha` (the padded row state collapses to the dense batch
    state) **then** compute the first batch round's univariate over the
    still-unfolded `eq_int`.

    This is the one round whose fold is row-shaped (no `eq_int` bind) while its
    sum is batch-shaped. `eq_int` rides through unchanged; the batch
    rounds bind it from the next round on."""
    planes = _bind_planes(planes, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _fix_last(planes: _Planes, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """The `fix_last` step: bind the final challenge and read off the four pair
    openings (the fully-folded length-1 state's single element). `_finalize_layer`
    fuses this into the whole-layer kernel -- a fold step lowering to one kernel is
    the fusion-by-construction rule, not just a speedup."""
    p = _bind_planes(planes, alpha)
    return p.n0[0], p.n1[0], p.d0[0], p.d1[0]


@cache
def _round_interp_constants(dtype: Any) -> tuple[Array, Array]:
    """Lagrange `naturals` ({0..DEGREE}) and the inverse-Vandermonde, hoisted once
    per dtype. Both depend only on `_DEGREE`, so rebuilding them inside every
    `prove_jagged_layer` trace is pure redundant host work -- `compute_inv_vandermonde`
    is an O(DEGREE^2) numpy coefficient build, redone per GKR layer without the memo."""
    # Force concrete eval: `@cache` memoizes the result, so building it inside a
    # jit trace (the jit=True round zone) would cache a tracer that then escapes the
    # trace (UnexpectedTracerError). The constants are trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, dtype)
    return naturals, inv_vand
