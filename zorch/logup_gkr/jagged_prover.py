# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged LogUp-GKR prover: the materialized per-layer sumcheck.

The jagged sibling of `prover.GkrLayerRound`. A `JaggedGkrLayer` materializes
only `sum(row_counts)` of its virtual `2^(niv+nrv)` positions; every
non-materialized position holds the fold-neutral fraction (n=0, d=1), whose
LogUp summand `eq * (lam*(n0*d1 + n1*d0) + d0*d1)` collapses to just its eq
weight. The sumcheck therefore runs over the materialized arrays and adds the
virtual mass back in closed form: the eq weights of a full hypercube sum to
the product of the bound variables' eq factors (`pad_adj`), so the
correction per round is `pad_adj - eq_sum_materialized`.

Round polynomials travel in coefficient form, interpolated through
{0, 1, 1/2, b}: the summand carries the current variable's eq factor, whose
root `b = (1-z)/(1-2z)` is known to both sides, so a degree-3 round needs
only the materialized evaluations at {0, 1/2} plus `s(1) = claim - s(0)`
(Gruen, https://eprint.iacr.org/2024/108). Value-form on the natural domain
would need a third materialized evaluation per round.

Variables bind LSB-first (consecutive-pair fold): a jagged layer is
interaction-major, so the row LSB is the in-segment pair dimension and the
stride-2 fold never crosses a segment boundary once odd segments are
re-padded (the same `_segment_gather` machinery as the circuit transition).
Row variables fold first while their eq factor rides as the materialized
`eq_row` lookup; once rows are exhausted the accumulated row-eq residual
becomes the scalar `eq_adj` and the interaction variables fold densely. The
bound point is challenges reversed -- LSB-first binding makes the last
challenge the MSB -- so the carry convention (MSB-first point, child selector
appended last) matches the dense chain's.

Per-round shapes shrink and the gather layout changes round to round, so the
driver is a host-orchestrated Python loop over plain numeric bodies, not the
homogeneous `zorch.sumcheck` scan (see docs/conventions.md).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, export

from zorch.fusion import fused_region
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _pad_neutral,
    _prepad_folded,
    _segment_gather_np,
)
from zorch.logup_gkr.prover import Carry, fold_carry, logup_combine
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
)
from zorch.round import Round
from zorch.sumcheck.prover import (
    SUMCHECK_MARKER,
    SUMCHECK_MARKER_VERSION,
    _state_leaves,
    fold_pair,
)
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    Transcript,
    reinterpret_challenge,
    sample_challenge,
)

if TYPE_CHECKING:
    from pathlib import Path

    from zorch.round import ProverRound

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2), in coefficient form.
_DEGREE = 3


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "lam",
        "claim",
        "round_polys",
        "point",
        "numerator_0",
        "numerator_1",
        "denominator_0",
        "denominator_1",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class JaggedLayerProof:
    """One jagged GKR layer's sumcheck transcript: the batching challenge and
    opening claim the layer entered with (the per-layer anchors a consumer
    diffs first when a byte-match diverges mid-pyramid), the coefficient-form
    round polynomials, the bound point, and the final pair openings.

    A pytree (every field is an `Array`, like the dense `sumcheck.RoundMsg`) so
    it can be returned across a `jax.jit` boundary -- the per-layer jit the
    chained prover wraps each round in.

    `point` is retained for wire serialization despite being replay-derivable
    — `LayerProof.point` carries the rationale and the
    verifier-must-never-read rule."""

    lam: Array
    claim: Array
    round_polys: Array  # (num_variables, _DEGREE + 1), ascending coefficients
    point: Array  # the bound point, MSB-first (the sampled challenges reversed)
    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array


def _round_metadata(
    row_counts: tuple[int, ...], num_row_vars: int
) -> list[tuple[Array | None, Array, Array]]:
    """Per-round `(gather, col_index, pair_index)` for the row-variable phase.

    Round k folds the layout round k-1 left behind: odd segments pre-pad to
    even (`gather`; None when already even), then the stride-2 fold halves
    every segment. `col_index` maps each pair to its interaction and
    `pair_index` to its in-segment pair offset -- the eq_row lookup is
    segment-local because a jagged layer is interaction-major, so the row-eq
    factor is indexed per segment while `eq_int[col_index]` carries the
    interaction weight. All static, derived from the Python-int row counts.
    """
    # Build the whole schedule on the host, then commit it in ONE batched
    # device_put: every index array is tiny and static, so a per-round transfer
    # is pure dispatch overhead (~3 per round x every layer). The None gathers
    # (no re-pad) ride through device_put as empty pytree nodes.
    host_meta: list[tuple[np.ndarray | None, np.ndarray, np.ndarray]] = []
    counts = row_counts
    for _ in range(num_row_vars):
        padded, pairs = _prepad_folded(
            counts
        )  # the circuit's own prepad/fold recurrence
        col_index = np.repeat(np.arange(len(pairs), dtype=np.int32), pairs)
        pair_index = np.concatenate([np.arange(pc, dtype=np.int32) for pc in pairs])
        host_meta.append((_segment_gather_np(counts, padded), col_index, pair_index))
        counts = pairs
    return jax.device_put(host_meta)


def _flatten_meta(
    meta: list[tuple[Array | None, Array, Array]],
) -> tuple[list[Array], list[bool]]:
    """Flatten the per-round `(gather, col_index, pair_index)` schedule into a flat
    operand list for the marker, dropping the `None` gathers (rounds needing no
    re-pad). The returned `gather_present` mask is the static structure
    `_unflatten_meta` rebuilds the tuples by -- it carries no array data, so it
    stays a Python value rather than a marker operand."""
    present = [gather is not None for gather, _, _ in meta]
    ops: list[Array] = []
    for gather, col_index, pair_index in meta:
        if gather is not None:
            ops.append(gather)
        ops.extend((col_index, pair_index))
    return ops, present


def _unflatten_meta(
    ops: Sequence[Array], present: list[bool]
) -> list[tuple[Array | None, Array, Array]]:
    """Inverse of `_flatten_meta`: rebuild the per-round meta tuples from the flat
    marker operands and the static presence mask."""
    meta: list[tuple[Array | None, Array, Array]] = []
    p = 0
    for has_gather in present:
        gather = ops[p] if has_gather else None
        p += int(has_gather)
        meta.append((gather, ops[p], ops[p + 1]))
        p += 2
    return meta


def _bind_lsb(arr: Array, r: Array) -> Array:
    """Bind the LSB variable: stride-2 consecutive pairs fold via the shared
    `sumcheck.prover.fold_pair` -- `e0 + r*(e1 - e0)`. (The split is LSB/stride-2,
    distinct from `fold`/`split_halves`' contiguous MSB halves; only the scalar
    fold is shared.)"""
    return fold_pair(arr[0::2], arr[1::2], r)


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
    natural domain: Lagrange-evaluate it at `naturals` ({0..3}), then
    `inv_vand` maps values to coefficients. Both are hoisted by the caller
    -- they only depend on the degree.
    """
    dtype = claim.dtype
    one = jnp.ones((), dtype)
    correction = pad_adj - eq_sum
    s_zero = (eval_zero + correction * (one - z_cur)) * eq_adj
    s_half = (
        (eval_half + correction * jnp.array(4, dtype)) / jnp.array(8, dtype) * eq_adj
    )
    s_one = claim - s_zero
    b_root = (one - z_cur) / (one - jnp.array(2, dtype) * z_cur)
    xs = jnp.stack([jnp.zeros((), dtype), one, one / jnp.array(2, dtype), b_root])
    ys = jnp.stack([s_zero, s_one, s_half, jnp.zeros((), dtype)])
    lagrange = jax.vmap(compute_lagrange_basis, in_axes=(0, None))(naturals, xs)
    return jnp.dot(inv_vand, jnp.dot(lagrange, ys))


def _paired_sums(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_0: Array,
    eq_1: Array,
    lam: Array,
) -> tuple[Array, Array, Array]:
    """Materialized `(s(0), 8*s(1/2), eq mass)` over the stride-2 pairs.

    s(0) reads the even elements at their eq weight; the u=1/2 sum works on
    doubled values (`e0 + e1 = 2*e(1/2)` per factor, likewise eq), which
    `_round_coeffs` rescales. Both go through the shared `logup_combine` so
    the summand cannot drift from the verifier oracle's.
    """
    eval_zero = jnp.sum(
        logup_combine(lam, eq_0, n0[0::2], d1[0::2], n1[0::2], d0[0::2])
    )
    eq_h = eq_0 + eq_1
    eval_half = jnp.sum(
        logup_combine(
            lam,
            eq_h,
            n0[0::2] + n0[1::2],
            d1[0::2] + d1[1::2],
            n1[0::2] + n1[1::2],
            d0[0::2] + d0[1::2],
        )
    )
    return eval_zero, eval_half, jnp.sum(eq_h)


# Jit the eq-table build: its eager log-depth doubling dispatches ~2*nrv ops a
# layer -- the relaunch's biggest non-kernel eager site. One pjit instead.
# Recompile-free here: the point length is nrv/niv, bounded by the row/segment
# variable count (a fixed bracket across shards, unlike the data-sized planes).
_jit_expand_eq = jax.jit(expand_eq_to_hypercube)


def prove_jagged_layer(
    layer: JaggedGkrLayer,
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    *,
    challenge_limbs: int = 1,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """Run one jagged GKR layer's materialized sumcheck.

    `eval_point` is MSB-first over (interaction || row) variables; its length
    fixes the virtual row depth `nrv = len(eval_point) - niv`, which may
    exceed what the materialized row counts need -- the extra rounds fold
    saturated all-ones segments against re-padded neutral rows, exactly the
    virtual positions' values. Returns the bound point (MSB-first, i.e. the
    challenges reversed), the advanced transcript, and the proof.
    """
    niv = layer.num_interaction_variables
    nrv = eval_point.shape[0] - niv
    if nrv < 1:
        raise ValueError(
            f"eval_point must carry at least one row variable: got "
            f"{eval_point.shape[0]} coordinates for {niv} interaction variables"
        )
    if max(layer.row_counts) > 1 << nrv:
        raise ValueError(
            f"row count {max(layer.row_counts)} exceeds the virtual row space "
            f"2^{nrv}; the row-eq lookup would run out of bounds"
        )

    one = jnp.ones((), eval_point.dtype)
    eq_row = _jit_expand_eq(eval_point[niv:], one)
    eq_int = _jit_expand_eq(eval_point[:niv], one)
    n0, n1 = layer.numerator_0, layer.numerator_1
    d0, d1 = layer.denominator_0, layer.denominator_1
    meta = _round_metadata(layer.row_counts, nrv)
    naturals, inv_vand = _round_interp_constants(eval_point.dtype)

    state = _JaggedState(
        _Planes(n0, n1, d0, d1), eq_row, eq_int, eval_point, lam, claim
    )
    sched = _JaggedSchedule(
        meta, _InterpConsts(naturals, inv_vand), nrv, niv, challenge_limbs
    )
    # Mark only a DuplexTranscript with a dedicated-fusion permutation, mirroring
    # `sumcheck.prover.prove`: the marker threads the sponge's leaves as operands.
    # The marked path decomposes through the same `_run_jagged_rounds`, so it is
    # byte-identical to the plain loop -- a vendor codegens it register-resident
    # over the sparse layout from the `row_counts` attribute (zkx#544).
    if (
        isinstance(transcript, DuplexTranscript)
        and transcript.has_dedicated_fusion
        and not transcript.fs_on_host
    ):
        out = _prove_jagged_marked(layer, state, sched, transcript)
    else:
        out = _run_jagged_rounds_relaunch(
            state, sched, transcript, export_dispatch=True
        )
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-round jagged sumcheck loop, shared by the plain and marked paths so
    the `zorch.sumcheck` marker decomposes byte-identically. Returns the bound
    point (challenges reversed), the advanced transcript, the stacked round
    polynomials, and the four folded pair openings."""
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    naturals, inv_vand = sched.consts.naturals, sched.consts.inv_vand
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    point = eval_point
    polys: list[Array] = []
    challenges: list[Array] = []
    for rnd in range(nrv + niv):
        in_rows = rnd < nrv
        if in_rows:
            gather, col_index, pair_index = meta[rnd]
            n0, n1, d0, d1 = _pad_neutral(n0, n1, d0, d1, gather)
            w = eq_int[col_index]
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0,
                n1,
                d0,
                d1,
                eq_row[pair_index * 2] * w,
                eq_row[pair_index * 2 + 1] * w,
                lam,
            )
        else:
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0, n1, d0, d1, eq_int[0::2], eq_int[1::2], lam
            )
        poly = _round_coeffs(
            eval_zero,
            eval_half,
            eq_sum,
            eq_adj,
            pad_adj,
            point[-1],
            claim,
            naturals,
            inv_vand,
        )
        transcript = transcript.observe(poly)
        transcript, r = sample_challenge(transcript, claim.dtype, challenge_limbs)
        polys.append(poly)
        challenges.append(r)

        claim, pad_adj = _fold_scalars(poly, r, pad_adj, point[-1], one)
        n0, n1, d0, d1 = (_bind_lsb(a, r) for a in (n0, n1, d0, d1))
        if in_rows:
            eq_row = _bind_lsb(eq_row, r)
            if rnd == nrv - 1:
                # Rows exhausted: the accumulated row-eq product becomes the
                # scalar factor of every interaction round; pad_adj restarts
                # to track the interaction variables' own bound mass.
                eq_adj = pad_adj
                pad_adj = one
        else:
            eq_int = _bind_lsb(eq_int, r)
        point = point[:-1]

    return (
        jnp.stack(challenges[::-1]),
        transcript,
        jnp.stack(polys),
        n0[0],
        n1[0],
        d0[0],
        d1[0],
    )


# ===== host-relaunch sumcheck engine (ported from the issue327 prototype) =====
# Recompile-free symbolic-export round kernels + the host loop relaunching one
# per round; FS runs through the transcript (device or host per fs_on_host).


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["n0", "n1", "d0", "d1"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _Planes:
    """The four LogUp MLE planes (numerator_0/1, denominator_0/1) as one pytree --
    they travel and bind together through every round. A registered pytree so it
    crosses the `jax.export` boundary as a single structured operand."""

    n0: Array
    n1: Array
    d0: Array
    d1: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["eq_adj", "pad_adj", "z_cur", "claim", "lam"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _RoundScalars:
    """The per-round scalar inputs to the round univariate: the eq/pad bound-mass
    corrections (`eq_adj`/`pad_adj`), the current point coordinate `z_cur`, the
    running `claim`, and the LogUp RLC coefficient `lam`. Scalar operands of the
    exported round kernel (a registered pytree)."""

    eq_adj: Array
    pad_adj: Array
    z_cur: Array
    claim: Array
    lam: Array


@dataclass(frozen=True)
class _InterpConsts:
    """The Lagrange interpolation constants (the `{0..DEGREE}` natural domain and
    the inverse Vandermonde). They depend only on dtype, so the round kernels bake
    them in as closure constants -- NOT export operands, so not a pytree."""

    naturals: Array
    inv_vand: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["planes", "eq_row", "eq_int", "eval_point", "lam", "claim"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _JaggedState:
    """A jagged layer's sumcheck carry: the four MLE planes, the row/interaction
    eq tables, the bound point, the RLC `lam`, and the running `claim`. Bundled so
    the round-loop functions take one state instead of nine loose arrays -- the
    `(round, state, transcript)` shape `sumcheck.prove` and jagged-pcs's
    `_InnerState` already use."""

    planes: _Planes
    eq_row: Array
    eq_int: Array
    eval_point: Array
    lam: Array
    claim: Array


@dataclass(frozen=True)
class _JaggedSchedule:
    """A jagged layer's static round schedule: the per-round gather/index metadata,
    the interpolation constants, and the interaction/row variable counts plus the
    challenge limb count. Rides beside the state so the loop signatures stay
    `(state, schedule, transcript)` rather than a 16-positional-arg bag."""

    meta: list[tuple[Array | None, Array, Array]]
    consts: _InterpConsts
    nrv: int
    niv: int
    challenge_limbs: int


def _bind_planes(planes: _Planes, alpha: Array) -> _Planes:
    return _Planes(
        *(_bind_lsb(a, alpha) for a in (planes.n0, planes.n1, planes.d0, planes.d1))
    )


def _round_poly_int(
    planes: _Planes, eq_int: Array, scalars: _RoundScalars, consts: _InterpConsts
) -> Array:
    """The `sum_as_poly` step for the dense interaction phase: the round
    univariate from the current state, no fold (the entry kernel of a
    relaunch chain, before any challenge is bound).

    Fiat-Shamir-less by construction — the host observes the returned poly and
    samples the challenge between launches — so the body is pure field
    arithmetic and exports portably at a symbolic state size. `eq_int` is sliced
    stride-2 once inside `_paired_sums`, so the state size is symbolic over an
    even bracket (`2*g`)."""
    eval_zero, eval_half, eq_sum = _paired_sums(
        planes.n0,
        planes.n1,
        planes.d0,
        planes.d1,
        eq_int[0::2],
        eq_int[1::2],
        scalars.lam,
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
    """The `fix_and_sum` step for the dense interaction phase: bind the previous
    round's challenge `alpha` (state size `m -> m/2`) **then** compute the next
    round's univariate, one relaunch of a single device kernel at the halved
    size. Returns `(poly, planes, eq_int)` so the host threads the folded state
    into the next launch and only the scalar poly crosses back up.

    The fold and the inner `_paired_sums` slice stride-2 twice, so the state size
    is symbolic over a multiple-of-four bracket (`4*g`): declaring the dim as
    `4*g` keeps both halvings decidable for `jax.export` (a free symbol leaves
    the parity inconclusive)."""
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
) -> tuple[Array, _Planes]:
    """The row-variable round body shared by the row kernels: re-pad the four
    MLEs to the round's even layout (`gather`), look the per-pair interaction eq
    weight up via `eq_int[col_index]`, and form the round univariate over the
    segment-local `eq_row` pairs. Returns `(poly, planes)` — the padded state the
    caller binds next round.

    The schedule (`gather`, `col_index`, `pair_index`) is a host-built operand,
    resolved with runtime indexing, so the body exports at a symbolic pair count.
    The post-pad state is `gather`'s length, declared `2*p` so the `_paired_sums`
    stride-2 stays decidable."""
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
    end. The input state and `eq_row` enter even (`2*…`), the `_pad_neutral`
    output is `2*p`; all halvings stay decidable for `jax.export`."""
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
    """The row->interaction handoff in one launch: bind the last row variable's
    challenge `alpha` (the padded row state collapses to the dense interaction
    state) **then** compute the first interaction round's univariate over the
    still-unfolded `eq_int`.

    This is the one round whose fold is row-shaped (no `eq_int` bind) while its
    sum is interaction-shaped. `eq_int` rides through unchanged; the interaction
    rounds bind it from the next launch on."""
    planes = _bind_planes(planes, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _fix_last(planes: _Planes, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """The `fix_last` step: bind the final challenge and read off the four pair
    openings (the fully-folded length-1 state's single element). The relaunch tail
    fuses this into one kernel via `_jit_finalize` -- a fold step lowering to one
    kernel is the fusion-by-construction rule, not just a speedup."""
    p = _bind_planes(planes, alpha)
    return p.n0[0], p.n1[0], p.d0[0], p.d1[0]


@cache
def _round_interp_constants(dtype: Any) -> tuple[Array, Array]:
    """Lagrange `naturals` ({0..DEGREE}) and the inverse-Vandermonde, hoisted once
    per dtype. Both depend only on `_DEGREE`, so rebuilding them inside every
    `prove_jagged_layer` is pure redundant host work -- `compute_inv_vandermonde`
    is an O(DEGREE^2) numpy coefficient build, and the per-GKR-layer recompute
    dominated the eager host glue around the relaunch loop."""
    # Force concrete eval: `@cache` memoizes the result, so building it inside a
    # jit trace (the jit=True round zone) would cache a tracer that then escapes the
    # trace (UnexpectedTracerError). The constants are trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, dtype)
    return naturals, inv_vand


# Exported per-round kernels, keyed by the operand signature so one binary
# serves every round size in its bracket and is reused across rounds, layers, and
# shards (the recompile-free relaunch). Only the per-round-REPEATED variants are
# cached here -- `fix_and_sum_row` (the row rounds) and `fix_and_sum_int` (the
# interaction rounds); `sum_as_poly_row` / `_boundary` / `_last` fire once a layer
# and stay eager. The state dtype is part of the key: a multi-limb sumcheck folds
# base->extension after round 0, so the row binary is dispatched at two input
# dtypes (numerator base-field, denominator extension-field).
#
# The bare `Exported` is cached, not `jax.jit(exported.call)`: jit-wrapping cuts
# the per-call host dispatch (cached exec vs bare's per-call re-specialize) but is
# wall-clock NEUTRAL on the real prove -- the round dispatch overlaps async GPU/FS
# work, so the saved host time is off the critical path. Not worth the extra layer.
_ROUND_KERNEL_CACHE: dict[tuple, export.Exported] = {}

# Opt-in on-disk cache for the exported round binaries (set ZORCH_EXPORT_CACHE_DIR):
# their jax.export BUILD (symbolic StableHLO generation) re-runs every process and
# dominates the cold start, which the XLA persistent compile cache does NOT cover.
# Namespaced by jax version + a hash of every module the exported kernels close
# over (see `_export_cache_dir`), so any kernel-arithmetic edit invalidates it.
# Unset -> unchanged in-memory behaviour.
_EXPORT_CACHE_DIR = os.environ.get("ZORCH_EXPORT_CACHE_DIR")


@cache
def _export_cache_dir() -> Path:
    import hashlib
    from pathlib import Path

    import zorch.logup_gkr.circuit as _circuit
    import zorch.logup_gkr.prover as _prover
    import zorch.poly.eq as _eq
    import zorch.poly.univariate as _univariate

    # Hash every module whose code the exported round kernels close over, so any
    # edit to the round arithmetic invalidates the on-disk binary: this module, the
    # eq / circuit plane builders, `logup_combine` (the summand) in logup_gkr.prover,
    # and the Lagrange/Vandermonde interpolation in poly.univariate. Miss one and a
    # stale binary silently emits the OLD arithmetic — a wrong proof.
    h = hashlib.sha256()
    for mod in (_circuit, _eq, _prover, _univariate):
        src = mod.__file__
        assert src is not None  # imported modules always carry a source path
        h.update(Path(src).read_bytes())
    h.update(Path(__file__).read_bytes())
    d = (
        # Reached only under the `_EXPORT_CACHE_DIR is not None` guards in
        # `_round_get`/`_round_put`, so the env var is a real path string here.
        Path(_EXPORT_CACHE_DIR)  # type: ignore[arg-type]
        / f"{jax.__version__}-{h.hexdigest()[:12]}"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _export_path(key: tuple) -> Path:
    import hashlib

    return (
        _export_cache_dir()
        / f"{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.bin"
    )


def _round_get(key: tuple) -> export.Exported | None:
    exp = _ROUND_KERNEL_CACHE.get(key)
    if exp is None and _EXPORT_CACHE_DIR is not None:
        path = _export_path(key)
        if path.exists():
            exp = export.deserialize(bytearray(path.read_bytes()))
            _ROUND_KERNEL_CACHE[key] = exp
    return exp


def _round_put(key: tuple, exp: export.Exported) -> None:
    _ROUND_KERNEL_CACHE[key] = exp
    if _EXPORT_CACHE_DIR is not None:
        _export_path(key).write_bytes(bytes(exp.serialize()))


# `_Planes` / `_RoundScalars` cross the jax.export boundary as pytree operands;
# register their (empty -- no meta_fields) aux so `Exported.serialize()` can
# round-trip them for the on-disk cache above.
if _EXPORT_CACHE_DIR is not None:
    for _t in (_Planes, _RoundScalars):
        try:
            export.register_pytree_node_serialization(
                _t,
                serialized_name=f"zorch.logup_gkr.jagged_prover.{_t.__name__}",
                serialize_auxdata=lambda _a: b"",
                deserialize_auxdata=lambda _b: (),
            )
        except ValueError:
            # Idempotent across a re-import (importlib.reload): serialized_name is
            # a constant, so jax raises "Duplicate serialization registration".
            # The prior, identical registration is already live -- keep it.
            pass

# The symbolic bound only needs to *contain* every round size (`exported.call`
# re-specializes XLA codegen per concrete size regardless), but it MUST exceed the
# largest dispatched state: a row input is `2*pp` and an interaction input `4*g`,
# so the bound caps the provable layer at `2*_ROUND_SYM_MAX` / `4*_ROUND_SYM_MAX`
# elements. Hold it well above any trace's 2^(log rows) so no real shard overflows.
_ROUND_SYM_MAX = 1 << 30


_SCALAR_FIELDS = ("eq_adj", "pad_adj", "z_cur", "claim", "lam")


def _abst_scalars(scalars: _RoundScalars) -> _RoundScalars:
    return _RoundScalars(
        *(jax.ShapeDtypeStruct((), getattr(scalars, f).dtype) for f in _SCALAR_FIELDS)
    )


def _dispatch_fix_and_sum_int(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the dense interaction round through one cached binary symbolic
    over the halving state size. The round folds (`m -> m/2`) then `_paired_sums`
    slices stride-2 again (`m/2 -> m/4`), two halvings with no re-pad between, so
    the state is `4*g` to keep both decidable; `eq_int` halves with it.

    `exported.call` is a host dispatch; under a `jax.jit` trace the operands are
    tracers, so fall back to the eager kernel -- the jit compiles the round
    itself, the host-relaunch export being its alternative."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_int(planes, eq_int, alpha, scalars, consts)
    operands = (planes, eq_int, alpha, scalars)
    # Per-operand dtypes (a LogUp numerator is base-field, its denominator
    # extension-field, and the state promotes base->extension across rounds), so
    # each (round-shape, dtype-mix) gets its own binary; `consts` is baked in.
    key = (
        "int",
        tuple(l.dtype for l in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )
    exported = _round_get(key)
    if exported is None:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((4 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda p, e, al, sc: _fix_and_sum_int(p, e, al, sc, consts)  # noqa: E731
        exported = export.export(jax.jit(fn))(*abst)
        _round_put(key, exported)
    return exported.call(*operands)


def _dispatch_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the row->interaction handoff (bind the last row challenge `alpha`,
    then sum the first interaction round over the still-unfolded `eq_int`) through
    one cached binary. Mirrors `_dispatch_fix_and_sum_int` without the `eq_int`
    bind: the bind halves the state (`4*g -> 2*g`) and `eq_int` rides unfolded at
    `2*g` (= the post-bind state), so a single relaunch replaces the eager kernel."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_boundary(planes, eq_int, alpha, scalars, consts)
    operands = (planes, eq_int, alpha, scalars)
    key = (
        "boundary",
        tuple(l.dtype for l in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )
    exported = _round_get(key)
    if exported is None:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda p, e, al, sc: _fix_and_sum_boundary(
            p, e, al, sc, consts
        )  # noqa: E731
        exported = export.export(jax.jit(fn))(*abst)
        _round_put(key, exported)
    return exported.call(*operands)


def _dispatch_sum_as_poly_row(
    planes: _Planes,
    gather: Array | None,
    col_index: Array,
    pair_index: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
) -> tuple[Array, _Planes]:
    """Dispatch the round-0 sum (no fold, no challenge) through one cached binary
    symbolic over the RAW layer height (`h`), the re-pad layout (`2*p` gather / `p`
    schedule), and `eq_row` (`2*rr`); `eq_int` rides fixed. Mirrors
    `_dispatch_fix_and_sum_row` without the bind -- a single relaunch replaces the
    eager entry kernel. A round needing no re-pad (`gather` None) gets an identity
    gather over the full height (no `//2` fold) so it hits the same binary."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _round_poly_row(
            planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
        )
    if gather is None:
        gather = jnp.arange(planes.n0.shape[0], dtype=col_index.dtype)
    operands = (planes, gather, col_index, pair_index, eq_row, eq_int, scalars)
    key = (
        "sum0",
        tuple(l.dtype for l in jax.tree_util.tree_leaves(operands)),
        eq_int.shape,
        consts.naturals.shape[0],
    )
    exported = _round_get(key)
    if exported is None:
        h, p, rr = export.symbolic_shape(
            "h, p, rr",
            constraints=[
                "h >= 1",
                f"h <= {_ROUND_SYM_MAX}",
                "p >= 1",
                f"p <= {_ROUND_SYM_MAX}",
                "rr >= 1",
                f"rr <= {_ROUND_SYM_MAX}",
            ],
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((h,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * p,), gather.dtype),
            jax.ShapeDtypeStruct((p,), col_index.dtype),
            jax.ShapeDtypeStruct((p,), pair_index.dtype),
            jax.ShapeDtypeStruct((2 * rr,), eq_row.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda pl, ga, ci, pi, er, ei, sc: _round_poly_row(  # noqa: E731
            pl, ga, ci, pi, er, ei, sc, consts
        )
        exported = export.export(jax.jit(fn))(*abst)
        _round_put(key, exported)
    return exported.call(*operands)


def _dispatch_fix_and_sum_row(
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
    """Dispatch a jagged row round through one cached binary symbolic over the
    input state (`2*pp`), the re-pad layout (`2*p` gather / `p` schedule), and the
    halving `eq_row` (`2*rr`); `eq_int` rides at its fixed `2^niv` width. A round
    that needs no re-pad (gather `None`) gets an identity gather so it hits the
    same binary -- the identity pad is a no-op, so byte-identical."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _fix_and_sum_row(
            planes,
            eq_row,
            alpha,
            gather,
            col_index,
            pair_index,
            eq_int,
            scalars,
            consts,
        )
    if gather is None:
        gather = jnp.arange(planes.n0.shape[0] // 2, dtype=col_index.dtype)
    operands = (planes, eq_row, alpha, gather, col_index, pair_index, eq_int, scalars)
    key = (
        "row",
        tuple(l.dtype for l in jax.tree_util.tree_leaves(operands)),
        eq_int.shape,
        consts.naturals.shape[0],
    )
    exported = _round_get(key)
    if exported is None:
        pp, p, rr = export.symbolic_shape(
            "pp, p, rr",
            constraints=[
                "pp >= 1",
                f"pp <= {_ROUND_SYM_MAX}",
                "p >= 1",
                f"p <= {_ROUND_SYM_MAX}",
                "rr >= 1",
                f"rr <= {_ROUND_SYM_MAX}",
            ],
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((2 * pp,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * rr,), eq_row.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            jax.ShapeDtypeStruct((2 * p,), gather.dtype),
            jax.ShapeDtypeStruct((p,), col_index.dtype),
            jax.ShapeDtypeStruct((p,), pair_index.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
        )
        fn = lambda pl, er, al, ga, ci, pi, ei, sc: _fix_and_sum_row(  # noqa: E731
            pl, er, al, ga, ci, pi, ei, sc, consts
        )
        exported = export.export(jax.jit(fn))(*abst)
        _round_put(key, exported)
    return exported.call(*operands)


def _fold_scalars(
    poly: Array, r: Array, pad_adj: Array, z: Array, one: Array
) -> tuple[Array, Array]:
    """The per-round scalar fold: the next claim (round poly evaluated at `r`) and the
    updated pad-mass `pad_adj`. One source for both the oracle `_run_jagged_rounds`
    (which inlines it) and the relaunch `_scalar_reduce` (which jits it), so the two
    cannot drift out of byte-equality."""
    return eval_coeffs(poly, r), pad_adj * (z * r + (one - z) * (one - r))


@jax.jit
def _scalar_reduce(
    poly: Array, r: Array, pad_adj: Array, z_cur: Array, one: Array
) -> tuple[Array, Array]:
    """The per-round scalar fold as ONE jitted dispatch instead of ~11 eager device
    ops -- the per-op JAX machinery (bind/apply_primitive) dominates the warm wall, so
    collapsing the op count is the lever (the ops themselves are async-cheap)."""
    return _fold_scalars(poly, r, pad_adj, z_cur, one)


# One squeezed challenge's reshape/bitcast (`raw.view`) jitted -- shape-stable per
# (limbs, dtype), so the ~2 eager ops per FS hop collapse to one dispatch.
_jit_reinterpret = jax.jit(reinterpret_challenge, static_argnums=(1,))


# The layer tail in one jitted dispatch: the final fold (`_fix_last`) plus stacking
# the per-round challenge/poly lists. Folding `_fix_last` in here keeps the final
# fold one kernel without decorating the bare helper, and the always-length-2 fold
# adds no shape -- recompile-free (one trace per challenge dtype / per round count,
# bounded by nrv); the ~2*(nrv+niv) element broadcasts + four fold slices collapse
# to one dispatch.
def _finalize_layer(
    planes: _Planes, alpha: Array, chal: list[Array], poly: list[Array]
) -> tuple[Array, Array, Array, Array, Array, Array]:
    fn0, fn1, fd0, fd1 = _fix_last(planes, alpha)
    return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)


_jit_finalize = jax.jit(_finalize_layer)


def _run_jagged_rounds_relaunch(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
    *,
    export_dispatch: bool = False,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """Host-relaunch sibling of `_run_jagged_rounds`: the same per-layer
    sumcheck as a host loop relaunching one fold-then-compute kernel per round at
    the round's real (halving) state size, Fiat-Shamir on the host between
    launches. One `sum_as_poly` (round 0, no fold), one `fix_and_sum` per
    subsequent round (row / boundary / interaction variant by round index), one
    `fix_last`. Byte-identical to `_run_jagged_rounds` — it regroups the same
    helper calls across the host FS boundary (fold of round k, sum of round k+1,
    in one kernel) rather than changing any arithmetic.

    Restructuring the per-round compute into one shape-polymorphic kernel is what
    lets a single `jax.export` binary serve every round size in a power-of-2
    bracket (the recompile-free relaunch); dispatching the kernels eagerly here
    is byte-equal to dispatching the exported binary, so this is the reference
    the export test pins to."""
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    planes = state.planes
    consts = sched.consts
    # The dispatch and eager kernels share signatures, so select one per round.
    fix_row = _dispatch_fix_and_sum_row if export_dispatch else _fix_and_sum_row
    fix_int = _dispatch_fix_and_sum_int if export_dispatch else _fix_and_sum_int
    fix_boundary = (
        _dispatch_fix_and_sum_boundary if export_dispatch else _fix_and_sum_boundary
    )
    # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
    sum0 = _dispatch_sum_as_poly_row if export_dispatch else _round_poly_row
    polys: list[Array] = []
    challenges: list[Array] = []
    prev_r = one  # unused until the first fold (round 1)
    for rnd in range(nrv + niv):
        # z_cur is eval_point's coordinate for this round (round k binds -(k+1)).
        # `jnp.take` with a static index lowers a single-element gather to a buffer
        # view -- zero dispatch -- where `eval_point[-(rnd+1)]` costs a slice+squeeze.
        z_cur = jnp.take(eval_point, -(rnd + 1))
        scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
        if rnd == 0:
            gather, col_index, pair_index = meta[0]
            poly, planes = sum0(
                planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
            )
        elif rnd < nrv:
            gather, col_index, pair_index = meta[rnd]
            poly, planes, eq_row = fix_row(
                planes,
                eq_row,
                prev_r,
                gather,
                col_index,
                pair_index,
                eq_int,
                scalars,
                consts,
            )
        elif rnd == nrv:
            poly, planes, eq_int = fix_boundary(planes, eq_int, prev_r, scalars, consts)
        else:
            poly, planes, eq_int = fix_int(planes, eq_int, prev_r, scalars, consts)
        transcript, raw = transcript.observe_and_sample(poly, challenge_limbs)
        r = _jit_reinterpret(raw, claim.dtype)
        polys.append(poly)
        challenges.append(r)
        claim, pad_adj = _scalar_reduce(poly, r, pad_adj, z_cur, one)
        if rnd == nrv - 1:
            eq_adj = pad_adj
            pad_adj = one
        prev_r = r

    fn0, fn1, fd0, fd1, stacked_challenges, stacked_polys = _jit_finalize(
        planes, prev_r, challenges, polys
    )
    return (
        stacked_challenges,
        transcript,
        stacked_polys,
        fn0,
        fn1,
        fd0,
        fd1,
    )


def _prove_jagged_marked(
    layer: JaggedGkrLayer,
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: DuplexTranscript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """Wrap `_run_jagged_rounds` in the hash-agnostic `zorch.sumcheck` composite,
    Fiat-Shamir INSIDE, so the body is the *same* loop and the result is
    bit-identical to the plain path.

    The static per-round schedule (`meta`), the interpolation tables
    (`naturals`/`inv_vand`), and `eval_point` ride as **explicit operands** -- they
    are tracer-valued under `@jit`, and only *closed-over* tracers are rejected by
    `lax.composite`, so passing them positionally keeps the leading auto-lifted
    operands the round-constant set alone (the emitter parses those by position).
    The duplex sponge threads through as the five `DuplexState` leaves; the FS
    permutation rides as the nested `zorch.poseidon2` marker inside `sample_challenge`.
    `row_counts` rides as the `array<i64>` attribute the vendor bounds each
    segment's reduction with (zkx#544); `fold_order`/`poly_form` declare the
    jagged LSB / coefficient-form contract (the dense defaults are MSB / value).
    """
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    naturals, inv_vand = sched.consts.naturals, sched.consts.inv_vand
    challenge_limbs = sched.challenge_limbs
    perm, rate = transcript.permutation, transcript.rate
    leaves = _state_leaves(transcript.state)
    meta_ops, gather_present = _flatten_meta(meta)
    n_meta = len(meta_ops)

    def body(*operands: Array, **_attrs: object) -> tuple[Array, ...]:
        bn0, bn1, bd0, bd1, beq_row, beq_int, beval, blam, bclaim = operands[:9]
        idx = 9
        lv = operands[idx : idx + len(leaves)]
        idx += len(leaves)
        bmeta = _unflatten_meta(operands[idx : idx + n_meta], gather_present)
        idx += n_meta
        bnaturals, binv_vand = operands[idx], operands[idx + 1]
        challenges, t, polys, fn0, fn1, fd0, fd1 = _run_jagged_rounds(
            _JaggedState(
                _Planes(bn0, bn1, bd0, bd1), beq_row, beq_int, beval, blam, bclaim
            ),
            _JaggedSchedule(
                bmeta, _InterpConsts(bnaturals, binv_vand), nrv, niv, challenge_limbs
            ),
            DuplexTranscript(perm, rate, DuplexState(*lv)),
        )
        # `_run_jagged_rounds` types its transcript as the generic `Transcript`;
        # here it is the DuplexTranscript built just above, so read its leaves back.
        # Result order is the recognizer's contract, shared with the dense
        # `_prove_marked`: [folded][5 sponge leaves][round polys][challenges].
        leaves_out = _state_leaves(cast(DuplexTranscript, t).state)
        return (fn0, fn1, fd0, fd1, *leaves_out, polys, challenges)

    out = fused_region(
        body,
        n0,
        n1,
        d0,
        d1,
        eq_row,
        eq_int,
        eval_point,
        lam,
        claim,
        *leaves,
        *meta_ops,
        naturals,
        inv_vand,
        name=SUMCHECK_MARKER,
        version=SUMCHECK_MARKER_VERSION,
        degree=_DEGREE,
        num_vars=nrv + niv,
        num_factors=4,
        row_counts=np.asarray(layer.row_counts, dtype=np.int64),
        fold_order="lsb",
        poly_form="coefficient",
    )
    fn0, fn1, fd0, fd1, *out_leaves, polys, challenges = out
    t = DuplexTranscript(perm, rate, DuplexState(*out_leaves))
    return challenges, t, polys, fn0, fn1, fd0, fd1


def _prove_jagged_layer_round(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """One jagged GKR layer's carry reduction: sample the batching `lam`, prove
    the layer, observe the openings, and fold the carry with the child selector.

    A module-level function (not a method) so it carries no implicit `self`:
    `JaggedGkrLayerRound` holds only its layer and `jit=True` routes through the
    shared `_jagged_round_zone` below, so the chain can drop a round -- and free
    its layer -- the moment it builds the next (the one-live-layer release
    `ChainedJaggedProveTest` pins; a `self`-closure or a per-instance jit would
    defer it)."""
    num_eval, den_eval, eval_point = carry
    transcript, lam = sample_challenge(transcript, num_eval.dtype, challenge_limbs)
    claim = lam * num_eval + den_eval
    point, transcript, proof = prove_jagged_layer(
        layer, lam, claim, eval_point, transcript, challenge_limbs=challenge_limbs
    )
    n0, n1 = proof.numerator_0, proof.numerator_1
    d0, d1 = proof.denominator_0, proof.denominator_1
    transcript = transcript.observe(jnp.stack([n0, n1, d0, d1]))
    transcript, r = sample_challenge(transcript, num_eval.dtype, challenge_limbs)
    num_eval, den_eval, eval_point = fold_carry(n0, n1, d0, d1, point, r)
    return (num_eval, den_eval, eval_point), transcript, proof


# Shared by every `JaggedGkrLayerRound(jit=True)`, keyed on the layer's static
# schedule (`row_counts`, `challenge_limbs`) and the four planes' shapes. A
# per-instance `jax.jit` gives each round a private trace cache, so a consumer
# that rebuilds the chain every warm iteration -- the generator that keeps lazy
# one-live-layer release -- would re-trace every layer of the pyramid on each
# call. Routing through one module-level zone lets freshly built same-shape
# rounds reuse a single trace. `row_counts` rides as a static arg (a small int
# tuple, hashed host-side); the planes ride as traced args -- shape-keyed, with
# no per-dispatch device->host sync that value-keying large planes would cost
# (cf. the #177 permutation fix). The layer is rebuilt inside from planes +
# counts (its `__post_init__` is static shape checks), so the traced body --
# and its output -- is identical to the eager round; only the cache key changes.
@partial(jax.jit, static_argnums=(4, 5))
def _jagged_round_zone(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    row_counts: tuple[int, ...],
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    layer = JaggedGkrLayer(
        numerator_0, numerator_1, denominator_0, denominator_1, row_counts
    )
    return _prove_jagged_layer_round(layer, challenge_limbs, carry, transcript)


def _jagged_round_via_zone(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Split the (non-pytree) layer into `_jagged_round_zone`'s traced planes +
    static `row_counts`. Signature mirrors `_prove_jagged_layer_round` so
    `JaggedGkrLayerRound` partials over either with one code path."""
    return _jagged_round_zone(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        layer.row_counts,
        challenge_limbs,
        carry,
        transcript,
    )


class JaggedGkrLayerRound(Round):
    """Prove one jagged GKR layer; the chain of these (floor outward) is the
    jagged GKR prover, threading the same `(num_eval, den_eval, eval_point)`
    carry as the dense chain. `challenge_limbs` rides on the round because
    every challenge in the layer -- lam, the per-variable folds, and the
    child-selector r -- must come from the same squeeze rule.

    The shared head `prover.bind_output` works unchanged for a jagged output
    when `challenge_limbs == 1`; a consumer squeezing multi-limb challenges
    owns its binding glue.

    With `jit=True` the per-layer prove dispatches through the module-level
    `_jagged_round_zone`, which traces once per layer *shape* and reuses that
    trace across every same-shape round -- so a consumer that rebuilds the
    chain each warm iter (the generator giving lazy one-live-layer release)
    pays the per-layer trace + composite build once across all iters, not once
    per iter. The round itself holds only its layer (no per-instance jit, no
    self-closure), so the chain's release bound is untouched. The pyramid stays
    a host-orchestrated Python loop of these (one trace per layer, never one
    `jit` over the whole pyramid -- it does not fit at scale; see
    `prover.LogupSumcheckRound`).
    """

    def __init__(
        self, layer: JaggedGkrLayer, challenge_limbs: int = 1, *, jit: bool = False
    ) -> None:
        # `partial` closes over (layer, challenge_limbs), not `self`, so the
        # chain frees the round -- and its layer -- the moment it builds the
        # next. `jit=True` dispatches through the shared module-level zone, so
        # same-shape rounds reuse one trace instead of re-compiling per call.
        body = _jagged_round_via_zone if jit else _prove_jagged_layer_round
        self._call = partial(body, layer, challenge_limbs)

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


def prove_jagged_pyramid(
    layers: Sequence[JaggedGkrLayer],
    carry: Carry,
    transcript: Transcript,
    *,
    challenge_limbs: int = 1,
) -> tuple[Carry, Transcript, list[JaggedLayerProof]]:
    """Retired: the device-side Fiat-Shamir jagged pyramid (one `lax.scan` over the
    floor-outward layer chain) is the abandoned device-FS path -- its on-chip
    FS-reuse peel chain only existed to keep an on-device sponge scan-invariant, and
    Fiat-Shamir now runs on the host between kernel launches. Prove via the unrolled
    `ProveChain(JaggedGkrLayerRound(l) for l in layers)` on a host-FS transcript
    (`fs_on_host=True`) -- byte-identical, and the production path."""
    raise NotImplementedError(
        "prove_jagged_pyramid (the device-FS rolled pyramid) is retired; prove via "
        "the unrolled JaggedGkrLayerRound chain on a host-FS transcript "
        "(fs_on_host=True)"
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
