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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import (
    Transcript,
    reinterpret_challenge,
    sample_challenge,
)

if TYPE_CHECKING:
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
    eq_row = expand_eq_to_hypercube(eval_point[niv:], one)
    eq_int = expand_eq_to_hypercube(eval_point[:niv], one)
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
    # The host round loop runs one fold-then-compute kernel per round, Fiat-Shamir
    # through the transcript between them. Under the production outer jit
    # (`JaggedGkrLayerRound(jit=True)`) the whole loop traces into one program; the
    # whole-scan `zorch.sumcheck` megakernel was retired -- it never compiled at
    # real sizes (mirrors #332's drop of the dense megakernel).
    out = _run_jagged_rounds(state, sched, transcript)
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


# ===== per-round jagged sumcheck engine =====
# The per-round fold-then-compute kernels + the host loop running one per round;
# FS runs through the transcript between rounds. Under the production outer jit the
# whole loop traces into one program.


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["n0", "n1", "d0", "d1"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _Planes:
    """The four LogUp MLE planes (numerator_0/1, denominator_0/1) as one pytree --
    they travel and bind together through every round. A registered pytree so it
    crosses the per-layer jit boundary as a single structured operand."""

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
    running `claim`, and the LogUp RLC coefficient `lam`. Bundled as one pytree so
    the round kernels take a single scalar operand."""

    eq_adj: Array
    pad_adj: Array
    z_cur: Array
    claim: Array
    lam: Array


@dataclass(frozen=True)
class _InterpConsts:
    """The Lagrange interpolation constants (the `{0..DEGREE}` natural domain and
    the inverse Vandermonde). They depend only on dtype, so the round kernels take
    them as a plain closure-constant bundle (not a traced operand, so not a
    pytree)."""

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
    round loop, before any challenge is bound).

    Fiat-Shamir-less by construction — the host observes the returned poly and
    samples the challenge between rounds — so the body is pure field arithmetic.
    `eq_int` is sliced stride-2 once inside `_paired_sums`, over an even state."""
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
    round's univariate, one round's bind + sum in one traced step at the halved
    size. Returns `(poly, planes, eq_int)` so the loop threads the folded state
    into the next round and only the scalar poly crosses back up. The fold and the
    inner `_paired_sums` each slice stride-2, so the state halves twice per round."""
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
    resolved with runtime indexing. The post-pad state is `gather`'s length, even
    so the `_paired_sums` stride-2 fold stays aligned."""
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
    end. The input state and `eq_row` enter even, the `_pad_neutral` output even
    too, so every stride-2 halving stays aligned."""
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
    """The row->interaction handoff in one traced step: bind the last row
    variable's challenge `alpha` (the padded row state collapses to the dense
    interaction state) **then** compute the first interaction round's univariate
    over the still-unfolded `eq_int`.

    This is the one round whose fold is row-shaped (no `eq_int` bind) while its
    sum is interaction-shaped. `eq_int` rides through unchanged; the interaction
    rounds bind it from the next round on."""
    planes = _bind_planes(planes, alpha)
    poly = _round_poly_int(planes, eq_int, scalars, consts)
    return poly, planes, eq_int


def _fix_last(planes: _Planes, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """The `fix_last` step: bind the final challenge and read off the four pair
    openings (the fully-folded length-1 state's single element). The round-loop tail
    groups this via `_finalize_layer` -- a fold step lowering to one kernel is the
    fusion-by-construction rule under the outer jit, not just a speedup."""
    p = _bind_planes(planes, alpha)
    return p.n0[0], p.n1[0], p.d0[0], p.d1[0]


@cache
def _round_interp_constants(dtype: Any) -> tuple[Array, Array]:
    """Lagrange `naturals` ({0..DEGREE}) and the inverse-Vandermonde, hoisted once
    per dtype. Both depend only on `_DEGREE`, so rebuilding them inside every
    `prove_jagged_layer` is pure redundant host work -- `compute_inv_vandermonde`
    is an O(DEGREE^2) numpy coefficient build, and the per-GKR-layer recompute
    dominated the eager host glue around the round loop."""
    # Force concrete eval: `@cache` memoizes the result, so building it inside a
    # jit trace (the jit=True round zone) would cache a tracer that then escapes the
    # trace (UnexpectedTracerError). The constants are trace-independent anyway.
    with jax.ensure_compile_time_eval():
        naturals = jnp.stack([jnp.array(j, dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, dtype)
    return naturals, inv_vand


def _fold_scalars(
    poly: Array, r: Array, pad_adj: Array, z: Array, one: Array
) -> tuple[Array, Array]:
    """The per-round scalar fold: the next claim (round poly evaluated at `r`) and the
    updated pad-mass `pad_adj`. One source for both the round loop's
    `_reinterpret_and_reduce` and the inline reference oracle in the tests, so the
    two cannot drift out of byte-equality."""
    return eval_coeffs(poly, r), pad_adj * (z * r + (one - z) * (one - r))


def _reinterpret_and_reduce(
    raw: Array,
    poly: Array,
    pad_adj: Array,
    z_cur: Array,
    one: Array,
    eval_point: Array,
    pos: Array,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, Array]:
    """Reinterpret the squeezed challenge, fold the round scalars, and slice the
    next round's eval-point coordinate. Grouped into one helper so the round loop
    reads as three named steps the outer jit fuses: the challenge reshape/bitcast,
    the scalar fold, and the per-round `eval_point` slice (a plain `jnp.take` would
    be a real ~22us dispatch, NOT a buffer view). `pos` indexes this
    round's coordinate; the next is `pos - 1`, threaded device-resident so no
    per-round index round-trips the host. Returns the round challenge `r`, the next
    `claim`, `pad_adj`, the next round's `z_cur`, and the decremented `pos`."""
    r = reinterpret_challenge(raw, dtype)
    claim, pad_adj = _fold_scalars(poly, r, pad_adj, z_cur, one)
    # The last round's `pos_next` is -1 (a dead output -- no round consumes it);
    # clamp so the slice index is provably in-bounds rather than leaning on
    # `dynamic_slice`'s implicit index clamp. No-op for every live round (pos >= 1).
    pos_next = jnp.maximum(pos - 1, jnp.int32(0))
    z_next = jax.lax.dynamic_index_in_dim(eval_point, pos_next, keepdims=False)
    return r, claim, pad_adj, z_next, pos_next


# The layer tail: the final fold (`_fix_last`) plus stacking the per-round
# challenge/poly lists. Grouped so the tail is one named step the outer jit fuses.
def _finalize_layer(
    planes: _Planes, alpha: Array, chal: list[Array], poly: list[Array]
) -> tuple[Array, Array, Array, Array, Array, Array]:
    fn0, fn1, fd0, fd1 = _fix_last(planes, alpha)
    return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-layer jagged sumcheck as a host loop: one fold-then-compute kernel
    per round at the round's real (halving) state size, Fiat-Shamir through the
    transcript between rounds. One `_round_poly_row` (round 0, no fold), one
    `_fix_and_sum_*` per subsequent round (row / boundary / interaction variant by
    round index), one `_fix_last`. Each kernel regroups the fold of round k with
    the sum of round k+1 so it lowers to one fused kernel; under the production
    outer jit (`JaggedGkrLayerRound(jit=True)`) the whole loop traces into one
    program. Byte-matched against the inline reference oracle in the tests (same
    math without the per-round kernel regrouping)."""
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
    polys: list[Array] = []
    challenges: list[Array] = []
    prev_r = one  # unused until the first fold (round 1)
    # z_cur is eval_point's coordinate for round k (== eval_point[-(k+1)]). Rather
    # than a standalone `jnp.take` every round (a real ~22us gather dispatch, not a
    # free buffer view), the coordinate is threaded device-resident: round 0 reads
    # the last coordinate and each `_reinterpret_and_reduce` slices the next via a
    # decremented `pos`, riding the fold's dispatch instead of its own. The fold stays
    # on the compute device (a host CPU reduce forces the carry to round-trip back to
    # GPU before each bind, which serializes the bind pipeline -- net slower).
    pos = jnp.asarray(eval_point.shape[0] - 1, jnp.int32)
    z_cur = jnp.take(eval_point, -1)
    for rnd in range(nrv + niv):
        scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
        if rnd == 0:
            # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
            gather, col_index, pair_index = meta[0]
            poly, planes = _round_poly_row(
                planes, gather, col_index, pair_index, eq_row, eq_int, scalars, consts
            )
        elif rnd < nrv:
            gather, col_index, pair_index = meta[rnd]
            poly, planes, eq_row = _fix_and_sum_row(
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
            poly, planes, eq_int = _fix_and_sum_boundary(
                planes, eq_int, prev_r, scalars, consts
            )
        else:
            poly, planes, eq_int = _fix_and_sum_int(
                planes, eq_int, prev_r, scalars, consts
            )
        transcript, raw = transcript.observe_and_sample(poly, challenge_limbs)
        r, claim, pad_adj, z_cur, pos = _reinterpret_and_reduce(
            raw, poly, pad_adj, z_cur, one, eval_point, pos, claim.dtype
        )
        polys.append(poly)
        challenges.append(r)
        if rnd == nrv - 1:
            eq_adj = pad_adj
            pad_adj = one
        prev_r = r

    fn0, fn1, fd0, fd1, stacked_challenges, stacked_polys = _finalize_layer(
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
