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
batch-major, so the row LSB is the in-segment pair dimension and the
stride-2 fold never crosses a segment boundary once odd segments are
re-padded (the same `_segment_gather` machinery as the circuit transition).
Row variables fold first while their eq factor rides as the materialized
`eq_row` lookup; once rows are exhausted the accumulated row-eq residual
becomes the scalar `eq_adj` and the batch variables fold densely. The
bound point is challenges reversed -- LSB-first binding makes the last
challenge the MSB -- so the carry convention (MSB-first point, child selector
appended last) matches the dense chain's.

Per-round shapes shrink and the gather layout changes round to round, so the
driver is a host-orchestrated Python loop over plain numeric bodies, not the
homogeneous `zorch.sumcheck` scan (see docs/conventions.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr._jagged_blocks import (
    _ROUND_BLOCK_SIZES,
    _dispatch_boundary_block,
    _dispatch_int_block,
    _dispatch_row_block,
)
from zorch.logup_gkr._jagged_buffers import (
    _pad_to_width,
    _pool_lay_batch,
    _resize_zero,
)
from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_fix_last,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_export import (
    _dispatch_fix_and_sum_boundary,
    _dispatch_fix_and_sum_int,
    _dispatch_fix_and_sum_row,
    _dispatch_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_fs import (
    _FS_EVAL_POINT_CAP,
    _fold_scalars,
    _fs_reduce_dispatch,
)
from zorch.logup_gkr._jagged_rounds import (
    _bind_lsb,
    _expand_eq_slice,
    _paired_sums,
    _round_coeffs,
    _round_interp_constants,
)
from zorch.logup_gkr._jagged_schedule import (
    _check_row_space,
    _dense_live_block,
    _dense_live_operand,
    _round_live_meta,
    _round_out_pairs,
    _row_counts_operand,
    _row_live_block,
    _row_live_blocks,
)
from zorch.logup_gkr._jagged_types import (
    RoundWidthCaps,
    _InterpConsts,
    _JaggedSchedule,
    _JaggedState,
    _Planes,
    _RoundScalars,
)
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _pad_neutral,
    _segment_gather,
)
from zorch.logup_gkr.prover import Carry, fold_carry
from zorch.round import Round
from zorch.transcript import (
    DuplexTranscript,
    Transcript,
    reinterpret_challenge,
    sample_challenge,
)

if TYPE_CHECKING:
    from zorch.round import ProverRound


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


def pad_layer_to_capacity(
    layer: JaggedGkrLayer, capacities: tuple[int, ...]
) -> JaggedGkrLayer:
    """Re-store `layer` in a capacity layout: each segment extended to its
    capacity with the fold-neutral fraction (n=0, d=1), and `row_counts`
    becoming the capacity tuple.

    The prove over the capacity layer is byte-identical to the exact layout:
    the neutral fraction is a fixed point of the per-round fold (and of the
    re-pad gathers a non-even capacity's schedule inserts), and its eq mass
    moves from the closed-form virtual correction (`pad_adj - eq_sum`) into
    the materialized sum -- the round polynomials, challenges, and openings do
    not change. What changes is the compile-key surface: the whole-layer
    program's plane and schedule shapes now derive from `capacities` alone, so
    shards sharing a capacity tuple share every trace and executable, and the
    true row counts ride only in this one gather's runtime data.

    Any `capacities >= row_counts` works; the choice trades memory against
    cache hits. A memory-tight consumer keeps capacities at a running
    per-segment max over its shards (padding ~= the inter-shard spread); a
    power-of-two capacity additionally makes every fold even (no per-round
    re-pad gathers) at up to 2x padding."""
    if len(capacities) != len(layer.row_counts):
        raise ValueError(
            f"capacities {capacities} must have one entry per segment "
            f"({len(layer.row_counts)})"
        )
    for rc, cap in zip(layer.row_counts, capacities, strict=True):
        if cap < rc:
            raise ValueError(f"capacity {cap} < row count {rc}")
    gather = _segment_gather(layer.row_counts, capacities)
    n0, n1, d0, d1 = _pad_neutral(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        gather,
    )
    return JaggedGkrLayer(n0, n1, d0, d1, capacities)


def prove_jagged_layer(
    layer: JaggedGkrLayer,
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    *,
    challenge_limbs: int = 1,
    caps: RoundWidthCaps | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """Run one jagged GKR layer's materialized sumcheck.

    `eval_point` is MSB-first over (batch || row) variables; its length
    fixes the virtual row depth `nrv = len(eval_point) - niv`, which may
    exceed what the materialized row counts need -- the extra rounds fold
    saturated all-ones segments against re-padded neutral rows, exactly the
    virtual positions' values. Returns the bound point (MSB-first, i.e. the
    challenges reversed), the advanced transcript, and the proof.

    `caps` selects the fixed-width round layout (xla#179 size-invariance):
    every round then runs at one static operand shape per phase, live prefix
    tracked by the rounds' `live` operand, so one compiled round kernel serves
    every round -- and every layer and shard proved under the same caps.
    Byte-identical to the exact layout.
    """
    niv = layer.num_batch_variables
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_from_counts(
        planes,
        niv,
        layer.row_counts,
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
    )


def _prove_jagged_layer_from_counts(
    planes: _Planes,
    niv: int,
    row_counts: tuple[int, ...],
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    challenge_limbs: int,
    caps: RoundWidthCaps | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """One jagged layer's sumcheck from the layer's static `row_counts`.

    Marker v2 (xla#179 device-derived schedule): the per-round re-pad schedule
    is a pure function of `row_counts` + the round index and derives inside
    the claimed kernels (and the decompositions), so the loop carries only the
    tiny i32[nseg] `row_counts` operand plus per-round i32[3] live triples —
    the hundreds-of-MB host-built gather uploads (and their per-warm-pass
    rebuild/staging) are gone, and the whole-layer jit's HLO stays tiny
    without `row_counts` ever entering the jit key (both ride as operands)."""
    nrv = eval_point.shape[0] - niv
    return _prove_jagged_layer_from_ops(
        planes,
        niv,
        _row_counts_operand(row_counts),
        _round_live_meta(row_counts, nrv),
        None if caps is not None else _round_out_pairs(row_counts, nrv),
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
        counts=row_counts,
    )


def _prove_jagged_layer_from_ops(
    planes: _Planes,
    niv: int,
    row_counts: Array,
    live: list[Array],
    out_pairs: tuple[int, ...] | None,
    lam: Array,
    claim: Array,
    eval_point: Array,
    transcript: Transcript,
    challenge_limbs: int,
    caps: RoundWidthCaps | None = None,
    counts: tuple[int, ...] | None = None,
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """`_prove_jagged_layer_from_counts` from PREBUILT schedule operands — the
    seam the whole-layer jit zone routes through, so `row_counts` and the live
    triples ride as TRACED operands (never keying the jit) while `out_pairs`
    (the exact layout's static padded widths; None under caps) stays static
    like `niv`/`caps`. `counts` (the Python-int layout, eager path only --
    the jit zone must not carry it) keys the pre-staged live-block cache."""
    nrv = eval_point.shape[0] - niv
    eq_row = _expand_eq_slice(eval_point, niv, row=True)
    eq_int = _expand_eq_slice(eval_point, niv, row=False)
    naturals, inv_vand = _round_interp_constants(eval_point.dtype)

    state = _JaggedState(planes, eq_row, eq_int, eval_point, lam, claim)
    sched = _JaggedSchedule(
        row_counts,
        live,
        out_pairs,
        _InterpConsts(naturals, inv_vand),
        nrv,
        niv,
        challenge_limbs,
        caps,
        counts=counts,
    )
    # The host round loop runs one fold-then-compute kernel per round, the FS hop
    # + reduce dispatching between them. `export_dispatch=True` selects the cached
    # per-round `jax.export` binary, but it only fires when this layer runs OUTSIDE
    # an outer jit (`JaggedGkrLayerRound(jit=False)`): the operands are then concrete
    # arrays, so each round host-dispatches and releases its buffers, bounding peak
    # host RAM on wide shards. Under the production outer jit
    # (`JaggedGkrLayerRound(jit=True)`) the dispatch sees tracers and falls back to
    # the marked kernel, tracing the whole loop into one program (the whole-scan
    # `zorch.sumcheck` megakernel was retired -- it never compiled at real sizes,
    # mirroring #332's drop of the dense megakernel).
    out = _run_jagged_rounds(state, sched, transcript, export_dispatch=True)
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


def _run_jagged_rounds_reference(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The unrolled oracle for `_run_jagged_rounds`: the per-round jagged sumcheck
    written out with an explicit observe/sample per round. Returns the bound point
    (challenges reversed), the advanced transcript, the stacked round polynomials,
    and the four folded pair openings. The round runner must match this byte-for-byte.
    """
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    if sched.meta is None:
        raise ValueError(
            "the reference oracle needs the schedule's host-built explicit "
            "meta (_round_metadata) — the round loop's derived-schedule "
            "fields do not carry it"
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
            # The oracle runs the exact layout; the schedule's `live` operand
            # (the fixed-width prefix marker) is the production loop's concern.
            gather, col_index, pair_index, _live = meta[rnd]
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
                # scalar factor of every batch round; pad_adj restarts
                # to track the batch variables' own bound mass.
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


# The layer tail: the final fold (`_fix_last`) plus stacking the per-round
# challenge/poly lists. Folding `_fix_last` in here keeps the final fold in the
# whole-layer kernel without decorating the bare helper. The width-preserving
# round buffers leave the fully-folded state as the live length-2 prefix, so
# the tail slices it down before the final marker -- the final ABI stays the
# exact (2,) planes.
def _stack_rounds(chal: list[Array], poly: list[Array]) -> tuple[Array, Array]:
    """Stack the per-round challenge/poly lists into the proof's transcript
    order. Multi-round block segments ((k,) challenges / (k, DEGREE+1) polys)
    mixed with singles flatten in round order -- the challenge reversal
    composes segment reversal with an in-segment flip. Same elements the
    stacked form carries, concatenated instead of stacked."""
    rev = [c[::-1] if c.ndim else c[None] for c in reversed(chal)]
    rows = [p if p.ndim == 2 else p[None] for p in poly]
    return (
        rev[0] if len(rev) == 1 else jnp.concatenate(rev),
        rows[0] if len(rows) == 1 else jnp.concatenate(rows),
    )


# The eager block path's stacking, hoisted into a module-level jit zone: the
# per-segment reverses / expands / concatenates otherwise dispatch one tiny
# execution each per layer (~5). Keyed on the lists' pytree structure +
# shapes -- static per layout. jit is byte-transparent, so the flattened
# transcript is unchanged.
_stack_rounds_zone = jax.jit(_stack_rounds)


def _finalize_layer(
    planes: _Planes,
    alpha: Array,
    chal: list[Array],
    poly: list[Array],
    openings: tuple[Array, Array, Array, Array] | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    if openings is not None:
        # A final boundary block already folded fix_last in-trace.
        fn0, fn1, fd0, fd1 = openings
    else:
        head = _Planes(*(a[:2] for a in (planes.n0, planes.n1, planes.d0, planes.d1)))
        fn0, fn1, fd0, fd1 = _composite_fix_last(head, alpha)
    if all(c.ndim == 0 for c in chal):
        # Per-round entries only (the traced and single-round paths): the
        # original one-stack structure, byte-for-byte.
        return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)
    stack = (
        _stack_rounds if isinstance(chal[0], jax.core.Tracer) else _stack_rounds_zone
    )
    stacked_chal, stacked_poly = stack(chal, poly)
    return fn0, fn1, fd0, fd1, stacked_chal, stacked_poly


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
    *,
    export_dispatch: bool = False,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-layer device-FS sumcheck host loop: one fold-then-compute per round
    at the round's real (halving) state size, the Fiat-Shamir hop + reduce folded in
    per round. One `sum_as_poly` (round 0, no fold), one `fix_and_sum` per subsequent
    round (row / boundary / batch variant by round index), one `fix_last`.

    On the default path this runs under the consumer's whole-layer `jax.jit`: every
    round's compute + FS hop traces into one fused layer kernel (the per-round host
    dispatches collapse to one per layer).

    `export_dispatch` selects, per round, the cached `jax.export` binary
    (`_dispatch_*`, one symbolic-size kernel host-relaunched at the halving size)
    over the eager kernel. It only fires when this loop runs OUTSIDE an outer
    `jax.jit` -- the operands are then concrete arrays, not tracers, so
    `exported.call` host-dispatches each round (the FS hop + reduce dispatching
    eagerly between rounds) and releases its buffers before the next, bounding
    peak host RAM (the decoupled production path). Under the outer jit
    (`JaggedGkrLayerRound(jit=True)`) the dispatch sees tracers and falls back
    to the marked kernel, tracing the whole loop into one program. Both paths are
    byte-identical to the inline reference oracle in the tests (same math; the
    export path only regroups it across per-round host dispatches)."""
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    nrv, niv = sched.nrv, sched.niv
    row_counts = sched.row_counts
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    planes = state.planes
    consts = sched.consts
    transcript = cast(DuplexTranscript, transcript)

    # Fixed-width layout (xla#179): lay the state into the capped buffers once
    # at layer entry; every round then runs at one static shape per phase with
    # the live prefix riding the `live` operand, so one compiled round kernel
    # serves every round/layer/shard under the caps. The dead tails are zeros
    # here and never read (every consumer masks by `live`).
    caps = sched.caps
    # The boundary round's pre-laid eq operand (concrete capped path only) --
    # set by the batched layer-entry lay-in below, consumed at the
    # row->interaction handoff.
    eq_boundary_pre: Array | None = None
    if caps is not None:
        if caps.row % 4:
            raise ValueError(
                f"row cap {caps.row} must be a multiple of 4 (the boundary "
                "handoff binds then pairs the row-width state, two stride-2 "
                "halvings)"
            )
        if caps.eq_row % 2 or caps.interaction % 4:
            raise ValueError(
                f"eq_row cap {caps.eq_row} must be even and interaction cap "
                f"{caps.interaction} a multiple of 4 (each folds stride-2 "
                "through its rounds)"
            )
        if caps.eq_row < eq_row.shape[0]:
            raise ValueError(
                f"eq_row cap {caps.eq_row} cannot hold the layer's row-eq "
                f"table ({eq_row.shape[0]})"
            )
        if caps.interaction < eq_int.shape[0]:
            raise ValueError(
                f"interaction cap {caps.interaction} cannot hold the layer's "
                f"interaction-eq table ({eq_int.shape[0]})"
            )
        if not isinstance(planes.n0, jax.core.Tracer):
            # Concrete (decoupled) path: lay each layer into the pooled,
            # donated cap buffers -- prefix-only in-place writes instead of
            # fresh cap-wide materializations (see _LAYER_BUF_POOL) -- through
            # ONE batched dispatch per layer instead of one per role. The
            # boundary round's eq operand (eq_int's live prefix at half the
            # plane width) rides the same dispatch: the row rounds never fold
            # eq_int, so laying it at entry is value-identical to laying it at
            # the row->interaction handoff where it is consumed.
            entries = [
                (f, getattr(planes, f), caps.row) for f in ("n0", "n1", "d0", "d1")
            ]
            entries.append(("eq_row", eq_row, caps.eq_row))
            entries.append(("eq_int", eq_int, caps.interaction))
            if niv > 0:
                half = caps.row // 2
                src = eq_int if eq_int.shape[0] <= half else eq_int[:half]
                entries.append(("eq_boundary", src, half))
            laid = _pool_lay_batch(entries)
            planes = _Planes(*laid[:4])
            eq_row, eq_int = laid[4], laid[5]
            if niv > 0:
                eq_boundary_pre = laid[6]
        else:
            planes = _Planes(
                *(
                    _pad_to_width(a, caps.row, 0)
                    for a in (planes.n0, planes.n1, planes.d0, planes.d1)
                )
            )
            eq_row = _pad_to_width(eq_row, caps.eq_row, 0)
            eq_int = _pad_to_width(eq_int, caps.interaction, 0)
    # The dispatch and marked kernels share signatures, so select one per round.
    # Both routes emit the `zorch.sumcheck.round` marker (the dispatch inside its
    # exported binary): a recognizing emitter fuses each round, and an unclaimed
    # marker decomposes inline, byte-identical to the eager body.
    fix_row = (
        _dispatch_fix_and_sum_row if export_dispatch else _composite_fix_and_sum_row
    )
    fix_int = (
        _dispatch_fix_and_sum_int if export_dispatch else _composite_fix_and_sum_dense
    )
    fix_boundary = (
        _dispatch_fix_and_sum_boundary
        if export_dispatch
        else _composite_fix_and_sum_boundary
    )
    # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
    sum0 = _dispatch_sum_as_poly_row if export_dispatch else _composite_sum_as_poly_row
    polys: list[Array] = []
    challenges: list[Array] = []
    prev_r = one  # unused until the first fold (round 1)
    # z_cur is eval_point's coordinate for round k (== eval_point[-(k+1)]). Rather
    # than a standalone `jnp.take` every round (a real ~22us gather dispatch, not a
    # free buffer view), the coordinate is threaded device-resident: round 0 reads
    # the last coordinate and each `_reduce_body` slices the next via a
    # decremented `pos`, riding the fold's dispatch instead of its own. The fold stays
    # on the compute device (a host CPU reduce forces the carry to round-trip back to
    # GPU before each bind, which serializes the bind pipeline -- net slower).
    pos = jnp.asarray(eval_point.shape[0] - 1, jnp.int32)
    z_cur = jnp.take(eval_point, -1)
    # On the eager path, pad eval_point to the fixed cap so the FS-hop jit
    # zone keys on ONE shape across every layer of the pyramid (eval_point's
    # length grows per layer; per-layer zone compiles multiplied the cold pass
    # ~6x when measured). `pos` and `z_cur` were derived from the live length
    # above, and the zone's `dynamic_index` only ever reads pos < live —
    # value-identical.
    if (
        not isinstance(eval_point, jax.core.Tracer)
        and eval_point.shape[0] <= _FS_EVAL_POINT_CAP
    ):
        eval_point = _pad_to_width(eval_point, _FS_EVAL_POINT_CAP, 0)
    # Multi-round blocks fire only on the decoupled capped path: concrete
    # operands (outside any outer jit) and fixed-width buffers (`out_pairs is
    # None` exactly when the caps fix every round shape). The traced path
    # keeps the single-round structure -- the whole layer already fuses into
    # one program there, so a block would change nothing but the trace shape.
    block_sizes = (
        _ROUND_BLOCK_SIZES
        if export_dispatch
        and caps is not None
        and sched.out_pairs is None
        and not isinstance(planes.n0, jax.core.Tracer)
        else ()
    )
    rnd = 0
    openings: tuple[Array, Array, Array, Array] | None = None
    while rnd < nrv + niv:
        if block_sizes and rnd < nrv:
            # Greedy row blocks over [0, nrv): K rounds per bind, challenge
            # chained inside the binary; a block starting at 0 folds the
            # layer's round 0 (sum, no fold) in as its first iteration.
            k = next((n for n in block_sizes if rnd + n <= nrv), 0)
            if k:
                scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
                live_ops = (
                    _row_live_blocks(sched.counts, nrv, rnd, k)
                    if sched.counts is not None
                    else _row_live_block(sched.live, rnd, k)
                )
                (
                    poly,
                    r,
                    planes,
                    eq_row,
                    transcript,
                    claim,
                    pad_adj,
                    z_cur,
                    pos,
                    prev_r,
                ) = _dispatch_row_block(
                    planes,
                    eq_row,
                    prev_r,
                    row_counts,
                    eq_int,
                    scalars,
                    consts,
                    live_ops,
                    transcript,
                    eval_point,
                    pos,
                    challenge_limbs,
                    first=rnd == 0,
                )
                polys.append(poly)
                challenges.append(r)
                if rnd + k == nrv:
                    # The block covered round nrv-1: the row->boundary swap
                    # (eq_adj takes the row stretch's pad mass) applies here,
                    # exactly as the single-round tail below does it.
                    eq_adj = pad_adj
                    pad_adj = one
                rnd += k
                continue
        if block_sizes and rnd == nrv and niv > 0:
            # The boundary handoff plus the dense stretch through one
            # binary; when it reaches the layer's last round it also folds
            # fix_last, handing the pair openings to _finalize_layer.
            k = next((n for n in block_sizes if n <= niv), 0)
            if k:
                # caps is not None on every block path (block_sizes guard),
                # so the boundary eq operand was pre-laid by the batched
                # layer-entry dispatch (eq_int is not folded by the row
                # rounds, so the entry lay is value-identical).
                assert eq_boundary_pre is not None
                scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
                (
                    poly,
                    r,
                    planes,
                    eq_int,
                    transcript,
                    claim,
                    pad_adj,
                    z_cur,
                    pos,
                    prev_r,
                    opens,
                ) = _dispatch_boundary_block(
                    planes,
                    eq_boundary_pre,
                    eq_int,
                    prev_r,
                    scalars,
                    consts,
                    _dense_live_block(1 << (niv - 1), k),
                    transcript,
                    eval_point,
                    pos,
                    challenge_limbs,
                    final=k == niv,
                )
                if opens is not None:
                    openings = opens
                polys.append(poly)
                challenges.append(r)
                rnd += k
                continue
        if block_sizes and nrv < rnd:
            # Greedy dense continuation past a partial boundary block (only
            # when the dense stretch outruns the largest block): the first
            # covered round folds `1 << (niv - 1 - (rnd - nrv))` pairs,
            # halving per round inside the block.
            k = next((n for n in block_sizes if rnd + n <= nrv + niv), 0)
            if k:
                scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
                (
                    poly,
                    r,
                    planes,
                    eq_int,
                    transcript,
                    claim,
                    pad_adj,
                    z_cur,
                    pos,
                    prev_r,
                ) = _dispatch_int_block(
                    planes,
                    eq_int,
                    prev_r,
                    scalars,
                    consts,
                    _dense_live_block(1 << (niv - 1 - (rnd - nrv)), k),
                    transcript,
                    eval_point,
                    pos,
                    challenge_limbs,
                )
                polys.append(poly)
                challenges.append(r)
                rnd += k
                continue
        scalars = _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam)
        dtype = claim.dtype
        if rnd == 0:
            out_pairs = None if sched.out_pairs is None else sched.out_pairs[0]
            poly, planes = sum0(
                planes,
                row_counts,
                eq_row,
                eq_int,
                scalars,
                consts,
                sched.live[0],
                out_pairs,
            )
        elif rnd < nrv:
            out_pairs = None if sched.out_pairs is None else sched.out_pairs[rnd]
            poly, planes, eq_row = fix_row(
                planes,
                eq_row,
                prev_r,
                row_counts,
                eq_int,
                scalars,
                consts,
                sched.live[rnd],
                out_pairs,
            )
        elif rnd == nrv:
            # The handoff's live pairs: the row phase saturates every segment
            # to one row by construction (row counts <= 2^nrv), so the last
            # padded row layout is exactly two slots per interaction --
            # 2^(niv+1) live elements, 2^(niv-1) post-bind pairs.
            live = _dense_live_operand(1 << (niv - 1))
            # The boundary marker needs its eq operand at half its plane width
            # (the post-bind state), so the capped route reads a resized copy
            # -- the live 2^niv prefix always fits in row // 2 (the last row
            # layout, 2^(niv+1) slots, fits in the row cap). `eq_int` itself
            # rides through the handoff unchanged, at its own width, for the
            # interaction rounds below.
            if caps is None:
                eq_boundary = eq_int
            elif isinstance(eq_int, jax.core.Tracer):
                eq_boundary = _resize_zero(eq_int, caps.row // 2)
            else:
                # Pre-laid by the batched layer-entry dispatch (concrete
                # path); the old per-handoff `_resize_zero` wrote a fresh
                # caps.row//2 buffer that is mostly zero tail at a wide cap.
                assert eq_boundary_pre is not None
                eq_boundary = eq_boundary_pre
            poly, planes, _ = fix_boundary(
                planes, eq_boundary, prev_r, scalars, consts, live
            )
            if caps is not None:
                # The handoff halves [row] -> [row // 2]; the dense phase runs
                # at the interaction cap, so resize to it -- the live prefix
                # (2^niv elements <= caps.interaction, validated above via the
                # eq table) always survives.
                planes = _Planes(
                    *(
                        _resize_zero(a, caps.interaction)
                        for a in (planes.n0, planes.n1, planes.d0, planes.d1)
                    )
                )
        else:
            live = _dense_live_operand(1 << (niv - 1 - (rnd - nrv)))
            poly, planes, eq_int = fix_int(
                planes, eq_int, prev_r, scalars, consts, live
            )
        # Device FS hop + reduce -- traced into the whole-layer jit on the default
        # path (one fused region per round), dispatched through the cached
        # `_fs_reduce_zone` between rounds on the export path (one executable
        # per hop instead of a composite retrace + ~15 element dispatches).
        # Slices the next z_cur via the decremented `pos`, riding the fold's
        # dispatch instead of a standalone gather.
        transcript, r, claim, pad_adj, z_cur, pos = _fs_reduce_dispatch(
            poly, transcript, pad_adj, z_cur, eval_point, pos, challenge_limbs, dtype
        )
        polys.append(poly)
        challenges.append(r)
        if rnd == nrv - 1:
            eq_adj = pad_adj
            pad_adj = one
        prev_r = r
        rnd += 1

    fn0, fn1, fd0, fd1, stacked_challenges, stacked_polys = _finalize_layer(
        planes, prev_r, challenges, polys, openings
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


# The inter-layer carry: sample `lam` + the batched claim before the round loop,
# absorb the openings + sample + fold the child selector after. Pure device math
# bracketing the layer's FS samples; traces into the whole-layer round zone.
def _sample_lam_and_claim(
    transcript: DuplexTranscript,
    num_eval: Array,
    den_eval: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array]:
    """The layer pre-carry: squeeze the batching `lam`, reinterpret it, and form the
    claim `lam*num_eval + den_eval`. All device math, traced into the whole-layer
    zone."""
    transcript, raw = transcript.sample(n)
    lam = reinterpret_challenge(raw, dtype)
    return transcript, lam, lam * num_eval + den_eval


def _observe_openings_and_fold(
    transcript: DuplexTranscript,
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    point: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array]:
    """Device-FS layer post-carry: absorb the four openings, squeeze the child
    selector `r`, and fold the carry. The openings stack, `observe`, `sample`,
    reinterpret, and `fold_carry` are all device math that trace into the whole-layer
    zone -- the layer-boundary sibling of the per-round FS hop. `observe_and_sample`
    fuses the absorb + squeeze exactly as the round FS does, so the transcript stream
    is byte-identical to the split form."""
    transcript, raw = transcript.observe_and_sample(jnp.stack([n0, n1, d0, d1]), n)
    r = reinterpret_challenge(raw, dtype)
    return transcript, *fold_carry(n0, n1, d0, d1, point, r)


def _prove_jagged_layer_round(
    planes: _Planes,
    niv: int,
    row_counts: Array,
    live: list[Array],
    out_pairs: tuple[int, ...] | None,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
    counts: tuple[int, ...] | None = None,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """One jagged GKR layer's carry reduction: sample the batching `lam`, prove
    the layer, observe the openings, and fold the carry with the child selector.

    Takes the planes + batch count + prebuilt schedule operands (not a
    `JaggedGkrLayer`) so the whole-layer jit never keys on `row_counts` and
    never bakes the schedule into the trace. A module-level function (no
    implicit `self`) so the chain can drop a round -- and free its layer --
    the moment it builds the next (the one-live-layer release
    `ChainedJaggedProveTest` pins)."""
    num_eval, den_eval, eval_point = carry
    dtype = num_eval.dtype
    transcript = cast(DuplexTranscript, transcript)
    # The per-layer carry brackets the round loop: sample lam + the batched claim
    # before, absorb the openings + sample + fold the child selector after. All
    # device math, traced into the whole-layer jit.
    transcript, lam, claim = _sample_lam_and_claim(
        transcript, num_eval, den_eval, challenge_limbs, dtype
    )
    point, transcript, proof = _prove_jagged_layer_from_ops(
        planes,
        niv,
        row_counts,
        live,
        out_pairs,
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
        counts=counts,
    )
    n0, n1 = proof.numerator_0, proof.numerator_1
    d0, d1 = proof.denominator_0, proof.denominator_1
    transcript, num_eval, den_eval, eval_point = _observe_openings_and_fold(
        cast(DuplexTranscript, transcript),
        n0,
        n1,
        d0,
        d1,
        point,
        challenge_limbs,
        dtype,
    )
    return (num_eval, den_eval, eval_point), transcript, proof


# Shared by every `JaggedGkrLayerRound(jit=True)`. The schedule operands
# (`row_counts` + the per-round live triples) ride as TRACED operands, not
# static args, so `row_counts` values leave the jit key: it keys only on the
# operand SHAPES plus the static `niv` / `challenge_limbs` / `caps` /
# `out_pairs` (`nrv` is read from `eval_point`'s length inside; `out_pairs` is
# None under caps, so the capped pyramid shares one key). Marker v2 shrank
# these operands from the hundreds-of-MB per-round gather arrays to KBs — the
# schedule now derives in-kernel — but the operand-not-closure rule stands:
# baking per-layer values into the trace would recompile per shard. Two layers
# still recompile when their shape sequence differs, but each compile is cheap
# and persistent-cached -- and under `caps` every layer shares ONE shape
# sequence, so the whole pyramid keys to a single trace. Routing through one
# module-level zone lets freshly built same-shape rounds reuse a single trace,
# so a consumer rebuilding the chain each warm iteration (the generator
# keeping lazy one-live-layer release) re-traces at most per distinct shape
# sequence, not per iter.
@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _jagged_round_zone(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    row_counts: Array,
    live: list[Array],
    niv: int,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    out_pairs: tuple[int, ...] | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    planes = _Planes(numerator_0, numerator_1, denominator_0, denominator_1)
    return _prove_jagged_layer_round(
        planes,
        niv,
        row_counts,
        live,
        out_pairs,
        challenge_limbs,
        caps,
        carry,
        transcript,
    )


def _jagged_round_via_zone(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Build the schedule operands host-side and dispatch through
    `_jagged_round_zone` with the planes + `row_counts` + live triples as
    traced operands. Splitting them out of the trace (rather than closing over
    the layer's static `row_counts`) is what keeps the whole-layer compile
    shard-independent."""
    niv = layer.num_batch_variables
    eval_point = carry[2]
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    return _jagged_round_zone(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        niv,
        challenge_limbs,
        caps,
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
        carry,
        transcript,
    )


def _jagged_round_eager(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """The `jit=False` body: build the schedule operands host-side and run the
    round loop eagerly -- each round (and its FS hop) dispatches on its own, so
    the export path can release every round's buffers before the next (the
    decoupled wide-shard production path)."""
    niv = layer.num_batch_variables
    eval_point = carry[2]
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_round(
        planes,
        niv,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
        challenge_limbs,
        caps,
        carry,
        transcript,
        counts=layer.row_counts,
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
    `_jagged_round_zone` with the round schedule as a traced operand, so it keys
    on `(niv, plane shapes)` and never on `row_counts` -- the trace (and its
    compiled kernel) is reused across every same-shape round, and shards
    differing only in row counts share one compile. With `jit=False` (default)
    the round loop runs eagerly -- each round's marked kernel (and the export
    dispatch, when it fires) releases its buffers before the next, bounding peak
    host RAM on wide shards. The round holds only its layer (no per-instance
    jit, no self-closure), so the chain's release bound is untouched. The
    pyramid stays a host-orchestrated Python loop of these (one trace per layer
    shape, never one `jit` over the whole pyramid -- it does not fit at scale;
    see `prover.LogupSumcheckRound`).
    """

    def __init__(
        self,
        layer: JaggedGkrLayer,
        challenge_limbs: int = 1,
        *,
        jit: bool = False,
        caps: RoundWidthCaps | None = None,
    ) -> None:
        # `partial` closes over (layer, challenge_limbs, caps), not `self`, so
        # the chain frees the round -- and its layer -- the moment it builds the
        # next. `jit=True` dispatches through the shared module-level zone, so
        # same-shape rounds reuse one trace instead of re-compiling per call.
        # `caps` selects the fixed-width round layout (see `prove_jagged_layer`).
        body = _jagged_round_via_zone if jit else _jagged_round_eager
        self._call = partial(body, layer, challenge_limbs, caps)

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
