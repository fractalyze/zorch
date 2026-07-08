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
import numpy as np
from jax import Array, lax

from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_fix_last,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_rounds import _paired_sums, _round_interp_constants
from zorch.logup_gkr._jagged_types import _JaggedState, _Planes, _RoundScalars
from zorch.logup_gkr.circuit import JaggedGkrLayer, _pad_neutral
from zorch.logup_gkr.prover import Carry, fold_carry
from zorch.round import Round
from zorch.sumcheck import gruen
from zorch.sumcheck.jagged.buffers import (
    _pad_to_width,
    _pool_lay_batch,
    _resize_zero,
)
from zorch.sumcheck.jagged.fs import _fs_reduce
from zorch.sumcheck.jagged.rounds import _expand_eq_slice, _round_coeffs
from zorch.sumcheck.jagged.schedule import (
    _check_row_space,
    _dense_live_operand,
    _fixed_width_gather,
    _round_live_meta,
    _round_out_pairs,
    _row_counts_operand,
)
from zorch.sumcheck.jagged.types import (
    RoundWidthCaps,
    _InterpConsts,
    _JaggedSchedule,
)
from zorch.sumcheck.prover import fold_lsb
from zorch.transcript import (
    DuplexState,
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
    """Prove one jagged GKR layer's materialized sumcheck from an explicit
    `lam` / `claim` (no inter-layer carry). The standalone single-layer seam
    the layer tests drive; the pyramid runs `JaggedGkrLayerRound`, which
    brackets this same core (`_prove_jagged_layer_from_ops`) with the carry.

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

    The device-derived schedule (xla#179): the per-round re-pad schedule is a
    pure function of `row_counts` + the round index and derives inside the
    claimed kernels, so the loop carries only the tiny i32[nseg] `row_counts`
    operand plus per-round i32[3] live triples -- both ride as traced operands,
    so `row_counts` never enters the jit key.
    """
    niv = layer.num_batch_variables
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    planes = _Planes(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return _prove_jagged_layer_from_ops(
        planes,
        niv,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
        lam,
        claim,
        eval_point,
        transcript,
        challenge_limbs,
        caps,
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
) -> tuple[Array, Transcript, JaggedLayerProof]:
    """One jagged layer's sumcheck from PREBUILT schedule operands — the shared
    core both entries reach: the whole-layer jit zone (`_prove_jagged_layer_round`)
    and the standalone `prove_jagged_layer`. `row_counts` and the live triples
    ride as TRACED operands (never keying the jit) while `out_pairs` (the exact
    layout's static padded widths; None under caps) stays static like
    `niv`/`caps`."""
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
    )
    # The host round loop runs one fold-then-compute kernel per round, the FS hop
    # + reduce traced between them. Under the whole-layer jit (`JaggedGkrLayerRound`)
    # the whole loop traces into one program (the whole-scan `zorch.sumcheck`
    # megakernel was retired -- it never compiled at real sizes, mirroring #332's
    # drop of the dense megakernel).
    out = _run_jagged_rounds(state, sched, transcript)
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


# The layer tail: the final fold (`_fix_last`) plus stacking the per-round
# challenge/poly lists. Folding `_fix_last` in here keeps the final fold in the
# whole-layer kernel without decorating the bare helper. The width-preserving
# round buffers leave the fully-folded state as the live length-2 prefix, so
# the tail slices it down before the final marker -- the final ABI stays the
# exact (2,) planes.
# Roll the row-phase rounds into a `lax.scan` once the row cap is at least this
# wide. A wide shard (rsp shard0/1/2, cap ~77.6M elems) unrolled co-residents its
# ~nrv capped `_Planes` buffers into a >20 GB XLA temp arena and OOMs a 32 GB card
# (sp1-zorch#254); the scan makes the round working set the loop carry, allocated
# once. A narrow shard (rsp shard17, cap ~4M) stays unrolled -- no scan, no perf
# regression where the unrolled arena already fit.
_ROW_SCAN_CAP = 1 << 24


def _finalize_layer(
    planes: _Planes,
    alpha: Array,
    chal: Array,
    poly: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """`chal`/`poly` are the per-round challenges/round-polys already stacked in
    round order; the bound point is the challenges reversed (LSB-first binding
    makes the last challenge the MSB)."""
    head = _Planes(*(a[:2] for a in (planes.n0, planes.n1, planes.d0, planes.d1)))
    fn0, fn1, fd0, fd1 = _composite_fix_last(head, alpha)
    return fn0, fn1, fd0, fd1, chal[::-1], poly


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
    *,
    force_row_scan: bool = False,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-layer device-FS sumcheck host loop: one fold-then-compute per round
    at the round's real (halving) state size, the Fiat-Shamir hop + reduce folded in
    per round. One `sum_as_poly` (round 0, no fold), one `fix_and_sum` per subsequent
    round (row / boundary / batch variant by round index), one `fix_last`.

    Runs under the consumer's whole-layer `jax.jit` (`JaggedGkrLayerRound`): every
    round's compute + FS hop traces into one fused layer kernel, so the per-round
    host dispatches collapse to one per layer. Each round emits the
    `zorch.sumcheck.round` marker (a recognizing emitter fuses it; an unclaimed
    marker decomposes inline, byte-identical to the eager body). Byte-identical to
    the inline reference oracle in the tests."""
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
    # here and never read (every consumer masks by `live`). The plane buffers
    # arrive already cap-width (the zone pre-lays them into the donated pool in
    # `_jagged_round_via_zone`), so `_pad_to_width` no-ops on them here.
    caps = sched.caps
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
        planes = _Planes(
            *(
                _pad_to_width(a, caps.row, 0)
                for a in (planes.n0, planes.n1, planes.d0, planes.d1)
            )
        )
        eq_row = _pad_to_width(eq_row, caps.eq_row, 0)
        eq_int = _pad_to_width(eq_int, caps.interaction, 0)
    fix_row = _composite_fix_and_sum_row
    fix_int = _composite_fix_and_sum_dense
    fix_boundary = _composite_fix_and_sum_boundary
    # Round 0 binds nothing yet, so its sum is the bare row poly (no fold).
    sum0 = _composite_sum_as_poly_row
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
    dtype = claim.dtype

    def _fs(
        poly: Array,
        transcript: DuplexTranscript,
        pad_adj: Array,
        z_cur: Array,
        pos: Array,
    ) -> tuple[DuplexTranscript, Array, Array, Array, Array, Array]:
        # Device FS hop + reduce -- traced into the whole-layer jit (one fused
        # region per round). Slices the next z_cur via the decremented `pos`,
        # riding the fold's dispatch instead of a standalone gather.
        return _fs_reduce(
            poly, transcript, pad_adj, z_cur, eval_point, pos, challenge_limbs, dtype
        )

    # Round 0 binds nothing yet: the bare row poly, no fold.
    out_pairs = None if sched.out_pairs is None else sched.out_pairs[0]
    poly, planes = sum0(
        planes,
        row_counts,
        eq_row,
        eq_int,
        _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam),
        consts,
        sched.live[0],
        out_pairs,
    )
    transcript, prev_r, claim, pad_adj, z_cur, pos = _fs(
        poly, transcript, pad_adj, z_cur, pos
    )
    polys.append(poly)
    challenges.append(prev_r)

    # Row-phase rounds 1..nrv-1: all `fix_row` at the row cap. Rounds 2..nrv-1
    # are uniform-shape under the caps, so a WIDE shard -- whose cap is large
    # enough that unrolling co-residents the ~nrv capped `_Planes` buffers into a
    # >20 GB XLA temp arena and OOMs (sp1-zorch#254) -- rolls them into a
    # `lax.scan`: the round working set becomes the loop carry, allocated once and
    # reused, still inside the outer @jit. Byte-identical (same per-round ops;
    # rows fold in the same order). A narrow shard keeps the unrolled path (no
    # scan overhead). `_RoundScalars` `eq_adj` is `one` throughout the row phase
    # (the residual is captured only at the handoff, below).
    row_polys = row_chals = None
    scan_rows = (
        nrv > 2 and caps is not None and (force_row_scan or caps.row > _ROW_SCAN_CAP)
    )

    def _fix_row_round(
        planes: _Planes,
        eq_row: Array,
        prev_r: Array,
        live_rnd: Array,
        out_pairs: int | None,
    ) -> tuple[Array, _Planes, Array]:
        poly, planes, eq_row = fix_row(
            planes,
            eq_row,
            prev_r,
            row_counts,
            eq_int,
            _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam),
            consts,
            live_rnd,
            out_pairs,
        )
        return poly, planes, eq_row

    if scan_rows:
        # Round 1 (the first fold) promotes the numerator planes BF -> EF; run it
        # explicitly so the scanned rounds 2..nrv-1 carry a loop-invariant (EF)
        # plane dtype (a `lax.scan` carry type must not change across iterations).
        poly, planes, eq_row = _fix_row_round(
            planes, eq_row, prev_r, sched.live[1], None
        )
        transcript, prev_r, claim, pad_adj, z_cur, pos = _fs(
            poly, transcript, pad_adj, z_cur, pos
        )
        polys.append(poly)
        challenges.append(prev_r)

        def _row_round(carry: Any, live_rnd: Array) -> tuple[Any, tuple[Array, Array]]:
            planes, eq_row, claim, pad_adj, z_cur, pos, prev_r, transcript = carry
            poly, planes, eq_row = fix_row(
                planes,
                eq_row,
                prev_r,
                row_counts,
                eq_int,
                _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam),
                consts,
                live_rnd,
                None,
            )
            transcript, r, claim, pad_adj, z_cur, pos = _fs(
                poly, transcript, pad_adj, z_cur, pos
            )
            return (
                planes,
                eq_row,
                claim,
                pad_adj,
                z_cur,
                pos,
                r,
                transcript,
            ), (poly, r)

        init = (planes, eq_row, claim, pad_adj, z_cur, pos, prev_r, transcript)
        carry, (row_polys, row_chals) = jax.lax.scan(
            _row_round, init, jnp.stack(sched.live[2:nrv])
        )
        planes, eq_row, claim, pad_adj, z_cur, pos, prev_r, transcript = carry
    else:
        for rnd in range(1, nrv):
            out_pairs = None if sched.out_pairs is None else sched.out_pairs[rnd]
            poly, planes, eq_row = _fix_row_round(
                planes, eq_row, prev_r, sched.live[rnd], out_pairs
            )
            transcript, prev_r, claim, pad_adj, z_cur, pos = _fs(
                poly, transcript, pad_adj, z_cur, pos
            )
            polys.append(poly)
            challenges.append(prev_r)

    # Rows exhausted: the accumulated row-eq residual becomes the scalar `eq_adj`
    # and the batch variables fold densely (the old rnd==nrv-1 arm, pulled out of
    # the loop so the row scan stays uniform).
    # After the row-phase branches join, re-narrow the transcript (the scan
    # branch unpacks it from an `Any` carry) so `_fs` keeps its DuplexTranscript.
    transcript = cast(DuplexTranscript, transcript)
    eq_adj = pad_adj
    pad_adj = one

    # Boundary round nrv, only when there are batch variables (niv == 0 has no
    # interaction phase -- the old `range(nrv + niv)` simply stopped after the
    # row rounds). The row -> interaction handoff: the marker needs its eq operand
    # at half its plane width (the post-bind state); the capped route reads a
    # resized copy (the live 2^niv prefix always fits in row // 2).
    if niv > 0:
        live = _dense_live_operand(1 << (niv - 1))
        if caps is None:
            eq_boundary = eq_int
        else:
            eq_boundary = _resize_zero(eq_int, caps.row // 2)
        poly, planes, _ = fix_boundary(
            planes,
            eq_boundary,
            prev_r,
            _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam),
            consts,
            live,
        )
        if caps is not None:
            # The handoff halves [row] -> [row // 2]; the dense phase runs at the
            # interaction cap, so resize to it -- the live prefix (2^niv elements
            # <= caps.interaction) always survives.
            planes = _Planes(
                *(
                    _resize_zero(a, caps.interaction)
                    for a in (planes.n0, planes.n1, planes.d0, planes.d1)
                )
            )
        transcript, prev_r, claim, pad_adj, z_cur, pos = _fs(
            poly, transcript, pad_adj, z_cur, pos
        )
        polys.append(poly)
        challenges.append(prev_r)

    # Dense (batch-variable) rounds nrv+1 .. nrv+niv-1.
    for rnd in range(nrv + 1, nrv + niv):
        live = _dense_live_operand(1 << (niv - 1 - (rnd - nrv)))
        poly, planes, eq_int = fix_int(
            planes,
            eq_int,
            prev_r,
            _RoundScalars(eq_adj, pad_adj, z_cur, claim, lam),
            consts,
            live,
        )
        transcript, prev_r, claim, pad_adj, z_cur, pos = _fs(
            poly, transcript, pad_adj, z_cur, pos
        )
        polys.append(poly)
        challenges.append(prev_r)

    # Stack the per-round polys/challenges in round order. A rolled row phase
    # (rounds 2..nrv-1) comes back stacked; splice it between the explicit head
    # (round 0 + the round-1 promotion = polys[:2]) and the boundary+dense tail
    # (polys[2:], empty when niv == 0).
    if row_polys is not None:
        segs_p = [jnp.stack(polys[:2]), row_polys]
        segs_c = [jnp.stack(challenges[:2]), row_chals]
        if len(polys) > 2:
            segs_p.append(jnp.stack(polys[2:]))
            segs_c.append(jnp.stack(challenges[2:]))
        stacked_polys = jnp.concatenate(segs_p)
        stacked_chals = jnp.concatenate(segs_c)
    else:
        stacked_polys = jnp.stack(polys)
        stacked_chals = jnp.stack(challenges)

    fn0, fn1, fd0, fd1, stacked_challenges, stacked_polys = _finalize_layer(
        planes, prev_r, stacked_chals, stacked_polys
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


# --- #254: fixed-max_rounds padded round body (FS-safe short-layer masking) ----
# The scan (a later task) runs a FIXED `max_rounds` loop so a whole pyramid of
# differently-sized layers shares one traced body. Layer j has only `nrv_j + niv`
# real rounds, so the surplus rounds must select the UNCHANGED carry -- the five
# transcript sponge leaves included -- or a short layer over-advances Fiat-Shamir
# (docs/conventions.md soundness rule; the chain is k separate sumchecks, not one
# joint scan, so this guard is what the homogeneous `zorch.sumcheck` scan omits).
# Isolated + tested on one layer here (no scan); the round body reuses HEAD's
# caps-width, `row_counts`-derived schedule (`_padded_round_schedule_jax`) and the
# exact LogUp round primitives (`_paired_sums` / `_round_coeffs` / `fold_lsb` /
# `gruen.fold_round_scalars`) the unrolled `_run_jagged_rounds_reference` oracle
# uses, so an active round is byte-identical to `_run_jagged_rounds`.

# The rolled step threads the per-round schedule through the marker as operands
# (the body rebuilds the dict from them in this fixed order); shared so producer
# and consumer agree.
_SCHED_KEYS = (
    "gather",
    "pair_lo",
    "pair_hi",
    "col",
    "live_pairs",
    "active",
    "in_rows",
    "last_row",
)


def _layer_plane_width(row_counts: tuple[int, ...], nrv: int, niv: int) -> int:
    """Fixed plane-buffer width for one layer: the widest per-round even-padded
    height (an odd segment padding up to even can exceed the layer's own height)
    or the dense interaction width, whichever is larger. The rolled pyramid pads
    every layer to the max of this across the chain so the scan stays
    shape-invariant. (Task 2 sizes the padded body at `caps.row` instead -- this
    is the scan-invariant sibling the schedule byte-match test pins.)"""
    width = 1 << niv
    counts = row_counts
    for _ in range(nrv):
        width = max(width, sum(rc + rc % 2 for rc in counts))
        counts = tuple((rc + rc % 2) // 2 for rc in counts)
    return width


def _excl_cumsum(x: Array) -> Array:
    """Exclusive prefix sum `out[i] = sum(x[:i])` via a broadcast mask -- `nseg`
    is tiny, so the n^2 cost is nil, and it dodges any `jnp.cumsum` fork quirk
    (the schedule reconstruction must lower cleanly under the scan)."""
    idx = jnp.arange(x.shape[0])
    return jnp.sum(jnp.where(idx[None, :] < idx[:, None], x[None, :], 0), axis=1)


def _padded_round_schedule(
    row_counts: tuple[int, ...],
    nrv: int,
    niv: int,
    max_rounds: int,
    plane_width: int,
) -> dict[str, np.ndarray]:
    """Host-side per-round schedule for one layer, padded to the fixed
    `max_rounds` / `plane_width`. The readable numpy oracle the runtime
    `_padded_round_schedule_jax` is byte-matched against.

    Each of `max_rounds` entries carries what the fixed-width round body reads off
    the static layout: the segment `gather` re-padding the plane buffer (indices
    `>= plane_width` resolve to the neutral fraction), the `eq_row` pair lookups
    (`pair_lo`/`pair_hi`) and their `eq_int` batch column (`col`), the live pair
    count the materialized sum is masked to (`live_pairs`), and the
    `active`/`in_rows`/`last_row` flags. Rounds past `nrv + niv` are INACTIVE (a
    shorter floor layer must not advance Fiat-Shamir there); rounds in
    `[nrv, nrv + niv)` are the dense interaction rounds (identity gather over the
    live prefix, the stride-2 `eq_int` split). Inactive / wrong-phase indices are
    in-bounds zeros; the live-pair mask and the `active` flag zero their
    contribution out."""
    half_width = plane_width // 2
    sentinel = plane_width  # `_gather_pad` treats any index >= width as padding

    gather = np.tile(np.arange(plane_width, dtype=np.int32), (max_rounds, 1))
    pair_lo = np.zeros((max_rounds, half_width), dtype=np.int32)
    pair_hi = np.zeros((max_rounds, half_width), dtype=np.int32)
    col = np.zeros((max_rounds, half_width), dtype=np.int32)
    live_pairs = np.zeros(max_rounds, dtype=np.int32)
    active = np.zeros(max_rounds, dtype=bool)
    in_rows = np.zeros(max_rounds, dtype=bool)
    last_row = np.zeros(max_rounds, dtype=bool)

    counts = row_counts
    for k in range(nrv):
        padded = tuple(rc + rc % 2 for rc in counts)
        pairs = tuple(p // 2 for p in padded)
        live = sum(padded) // 2
        gather[k] = _fixed_width_gather(counts, padded, plane_width)
        ci = np.repeat(np.arange(len(pairs), dtype=np.int32), pairs)
        pi = np.concatenate([np.arange(pc, dtype=np.int32) for pc in pairs])
        pair_lo[k, : pi.shape[0]] = pi * 2
        pair_hi[k, : pi.shape[0]] = pi * 2 + 1
        col[k, : ci.shape[0]] = ci
        live_pairs[k] = live
        active[k] = True
        in_rows[k] = True
        last_row[k] = k == nrv - 1
        counts = pairs

    for m in range(niv):
        k = nrv + m
        int_width = 1 << (niv - m)  # dense interaction width entering this round
        half = int_width // 2
        gather[k, :int_width] = np.arange(int_width, dtype=np.int32)
        gather[k, int_width:] = sentinel
        pair_lo[k, :half] = np.arange(half, dtype=np.int32) * 2
        pair_hi[k, :half] = np.arange(half, dtype=np.int32) * 2 + 1
        live_pairs[k] = half
        active[k] = True

    return {
        "gather": gather,
        "pair_lo": pair_lo,
        "pair_hi": pair_hi,
        "col": col,
        "live_pairs": live_pairs,
        "active": active,
        "in_rows": in_rows,
        "last_row": last_row,
    }


def _padded_round_schedule_jax(
    row_counts: Array,
    nrv: Array,
    niv: int,
    max_rounds: int,
    plane_width: int,
) -> dict[str, Array]:
    """`_padded_round_schedule` reconstructed from a RUNTIME `row_counts` (jax
    `i32[nseg]`) and runtime `nrv` -- the derived-schedule form (xla#179): no
    host-baked plane-width arrays, so the rolled scan computes its schedule from
    the compact row-count channel instead of stacking a multi-GB constant.
    Byte-identical to the numpy `_padded_round_schedule` (`PaddedRoundScheduleJaxTest`).

    The row counts halve each round and `ceil(ceil(x/2)/2 ...)` collapses to a
    single ceil-divide, so `counts_k = ceil(row_counts / 2^k)`. Every index pattern
    is then an iota plus segment offsets (`_segment_of` on the exclusive cumsum),
    so the only constants left are scalars."""
    nseg = row_counts.shape[0]
    half_width = plane_width // 2
    sentinel = plane_width
    i32 = jnp.int32
    p = jnp.arange(plane_width, dtype=i32)
    q = jnp.arange(half_width, dtype=i32)
    # Dense interaction widths 2^(niv-m) for m in [0, niv] -- a static table
    # indexed by the runtime m, dodging a runtime shift the fork may lack.
    int_table = jnp.asarray([1 << (niv - mm) for mm in range(niv + 1)], i32)

    def _segment_of(offsets: Array, pos: Array) -> Array:
        # The segment each position falls in: count of offsets <= pos, minus 1
        # (a broadcast `searchsorted(..., side="right") - 1` -- nseg is tiny, so
        # the n*nseg compare is nil and it avoids jnp.searchsorted's method flag).
        return jnp.clip(
            jnp.sum((offsets[None, :] <= pos[:, None]).astype(i32), axis=1) - 1,
            0,
            nseg - 1,
        )

    gather_l, pair_lo_l, pair_hi_l, col_l = [], [], [], []
    live_pairs_l, active_l, in_rows_l, last_row_l = [], [], [], []

    for k in range(max_rounds):
        in_rows_k = k < nrv
        active_k = k < nrv + niv

        # Row-round layout: counts_k = ceil(row_counts / 2^k); pad odd segs to
        # even, then halve. Segment offsets are exclusive cumsums.
        bk = 1 << k
        if bk > plane_width:
            # 2^k exceeds the buffer width, so every segment has folded to a
            # single element: ceil(row_counts / 2^k) == 1 (row_counts >= 1).
            # Saturate directly -- `row_counts + (bk - 1)` overflows int32 once
            # `bk` nears 2^31, reachable when max_rounds widens to the envelope.
            counts_k = (row_counts > 0).astype(i32)
        else:
            counts_k = (row_counts + (bk - 1)) // bk
        padded_k = counts_k + (counts_k % 2)
        pairs_k = padded_k // 2
        src_off = _excl_cumsum(counts_k)
        dst_off = _excl_cumsum(padded_k)
        pair_off = _excl_cumsum(pairs_k)
        total_dst = jnp.sum(padded_k)
        total_pairs = jnp.sum(pairs_k)

        seg_p = _segment_of(dst_off, p)
        within_p = p - dst_off[seg_p]
        live_p = (within_p < counts_k[seg_p]) & (p < total_dst)
        gather_row = jnp.where(live_p, src_off[seg_p] + within_p, sentinel).astype(i32)

        seg_q = _segment_of(pair_off, q)
        within_q = q - pair_off[seg_q]
        live_q = q < total_pairs
        col_row = jnp.where(live_q, seg_q, 0).astype(i32)
        pair_lo_row = jnp.where(live_q, within_q * 2, 0).astype(i32)
        pair_hi_row = jnp.where(live_q, within_q * 2 + 1, 0).astype(i32)

        # Interaction-round layout (dense): int_width = 2^(niv - (k - nrv)).
        m = jnp.clip(k - nrv, 0, niv)
        int_width = int_table[m]
        half = int_width // 2
        gather_int = jnp.where(p < int_width, p, sentinel).astype(i32)
        pair_lo_int = jnp.where(q < half, q * 2, 0).astype(i32)
        pair_hi_int = jnp.where(q < half, q * 2 + 1, 0).astype(i32)

        # Select row (in_rows) / interaction (active & ~in_rows) / inactive. An
        # inactive round keeps the identity gather (numpy's `tile(arange)` init)
        # and zeroed lookups; its `active` flag zeros the contribution anyway.
        gather_l.append(
            jnp.where(in_rows_k, gather_row, jnp.where(active_k, gather_int, p))
        )
        col_l.append(jnp.where(in_rows_k, col_row, 0).astype(i32))
        pair_lo_l.append(
            jnp.where(
                in_rows_k, pair_lo_row, jnp.where(active_k, pair_lo_int, 0)
            ).astype(i32)
        )
        pair_hi_l.append(
            jnp.where(
                in_rows_k, pair_hi_row, jnp.where(active_k, pair_hi_int, 0)
            ).astype(i32)
        )
        live_pairs_l.append(
            jnp.where(in_rows_k, total_pairs, jnp.where(active_k, half, 0)).astype(i32)
        )
        active_l.append(active_k)
        in_rows_l.append(in_rows_k)
        last_row_l.append(k == nrv - 1)

    return {
        "gather": jnp.stack(gather_l),
        "pair_lo": jnp.stack(pair_lo_l),
        "pair_hi": jnp.stack(pair_hi_l),
        "col": jnp.stack(col_l),
        "live_pairs": jnp.stack(live_pairs_l),
        "active": jnp.stack(active_l),
        "in_rows": jnp.stack(in_rows_l),
        "last_row": jnp.stack(last_row_l),
    }


def _select_active(active: Array, folded: Array, kept: Array) -> Array:
    """Fixed-width fold-back: a fold halves the live prefix, so re-pad it to the
    kept buffer's width before the select keeps it on active rounds (and the
    unchanged buffer otherwise)."""
    padded = jnp.concatenate([folded, jnp.zeros_like(folded)])[: kept.shape[0]]
    return jnp.where(active, padded, kept)


def _select_transcript(
    active: Array, advanced: Transcript, kept: Transcript
) -> Transcript:
    """Select the advanced sponge on active rounds, the unchanged one otherwise,
    leaf by leaf -- the transcript-neutral guard a shorter layer's padding rounds
    need. Both are `DuplexTranscript`s built off the same permutation/rate/fs."""
    a = cast(DuplexTranscript, advanced)
    k = cast(DuplexTranscript, kept)
    sel = lambda x, y: jnp.where(active, x, y)
    return a._with_state(
        DuplexState(
            sel(a.state.input_buffer, k.state.input_buffer),
            sel(a.state.output_buffer, k.state.output_buffer),
            sel(a.state.sponge_state, k.state.sponge_state),
            sel(a.state.in_pos, k.state.in_pos),
            sel(a.state.out_pos, k.state.out_pos),
        )
    )


def _run_jagged_rounds_padded(
    planes: _Planes,
    eq_row: Array,
    eq_int: Array,
    coords: Array,
    lam: Array,
    claim: Array,
    transcript: Transcript,
    sched: dict[str, Array],
    *,
    niv: int,
    max_rounds: int,
    real_rounds: Array,
    caps: RoundWidthCaps,
    challenge_limbs: int = 1,
) -> tuple[Array, Array, _Planes, Transcript]:
    """Fixed-width sibling of `_run_jagged_rounds`: run a FIXED `max_rounds` loop
    over caps-width neutral-padded buffers so a whole pyramid of differently-sized
    layers shares one traced round body. The four planes ride the `caps.row`
    buffer (live prefix at the front, neutral fraction n=0/d=1 in the tail); the
    per-round `sched` re-pads and folds inside that width and masks the
    materialized sum to the live pairs, so the dead tail never enters it (field
    zero-adds are exact -> byte-equal to `_run_jagged_rounds`'s live-only sum).
    `coords` is the bound point in round order (`eval_point` reversed).

    INACTIVE rounds (`sched["active"]` false -- a shorter layer's padding rounds
    past `nrv + niv`) select every carried value UNCHANGED, the transcript's five
    sponge leaves included, so the layer never over-advances Fiat-Shamir; the
    chain is k separate sumchecks, not one joint scan, so this guard is what the
    homogeneous `zorch.sumcheck` scan does not need.

    Returns `(polys, chal, planes_final, transcript)`: `polys` are the round
    polynomials stacked over `max_rounds` (surplus entries zero) in round order;
    `chal` is the bound point (the real challenges reversed, front-aligned into
    the fixed buffer via `real_rounds` so the caller slices `chal[:real_rounds]`);
    `planes_final` holds the four scalar pair openings."""
    del caps  # widths ride the arrays; kept for the caller/scan ABI
    dtype = claim.dtype
    naturals, inv_vand = _round_interp_constants(dtype)
    n0, n1, d0, d1 = planes.n0, planes.n1, planes.d0, planes.d1
    one = jnp.ones((), dtype)
    zero = jnp.zeros((), dtype)
    eq_adj = one
    pad_adj = one
    transcript = cast(DuplexTranscript, transcript)
    polys: list[Array] = []
    challenges: list[Array] = []

    for rnd in range(max_rounds):
        active = sched["active"][rnd]
        in_rows = sched["in_rows"][rnd]
        last_row = sched["last_row"][rnd]
        z_cur = coords[rnd]
        gather = sched["gather"][rnd]
        lo, hi, col = sched["pair_lo"][rnd], sched["pair_hi"][rnd], sched["col"][rnd]
        live_pairs = jnp.where(active, sched["live_pairs"][rnd], jnp.int32(0))

        pn0, pn1, pd0, pd1 = _pad_neutral(n0, n1, d0, d1, gather)
        # Row rounds weight each pair by its segment-local `eq_row` times its
        # interaction column `eq_int[col]`; interaction rounds use the dense
        # stride-2 `eq_int` split. Both weights are computed and `in_rows` selects
        # one; the discarded phase's indices can overrun its (smaller) eq table
        # (a row round's `pair_lo` overruns `eq_int`, an interaction round's
        # overruns `eq_row`), so clamp -- the live mask zeros the dead branch, and
        # the selected phase's indices are always in-bounds (clamp a no-op there).
        w = eq_int[jnp.minimum(col, eq_int.shape[0] - 1)]
        eq0_row = eq_row[jnp.minimum(lo, eq_row.shape[0] - 1)] * w
        eq1_row = eq_row[jnp.minimum(hi, eq_row.shape[0] - 1)] * w
        eq0_int = eq_int[jnp.minimum(lo, eq_int.shape[0] - 1)]
        eq1_int = eq_int[jnp.minimum(hi, eq_int.shape[0] - 1)]
        eq0 = jnp.where(in_rows, eq0_row, eq0_int)
        eq1 = jnp.where(in_rows, eq1_row, eq1_int)

        eval_zero, eval_half, eq_sum = _paired_sums(
            pn0, pn1, pd0, pd1, eq0, eq1, lam, live_pairs=live_pairs
        )
        poly = _round_coeffs(
            eval_zero,
            eval_half,
            eq_sum,
            eq_adj,
            pad_adj,
            z_cur,
            claim,
            naturals,
            inv_vand,
        )

        # Split FS hop (observe then squeeze) -- byte-identical to the fused
        # `_fs_reduce` (the runner reference gate pins this). Inactive rounds leave
        # the sponge (all five leaves) and the challenge untouched; the neutral
        # proof slot they emit is sliced off by the caller.
        observed = transcript.observe(poly)
        observed, r = sample_challenge(observed, dtype, challenge_limbs)
        transcript = _select_transcript(active, observed, transcript)
        r = jnp.where(active, r, zero)
        polys.append(jnp.where(active, poly, jnp.zeros_like(poly)))
        challenges.append(r)

        next_claim, next_pad = gruen.fold_round_scalars(poly, r, pad_adj, z_cur)
        fn0, fn1, fd0, fd1 = (fold_lsb(a, r) for a in (pn0, pn1, pd0, pd1))
        feq_row = fold_lsb(eq_row, r)

        claim = jnp.where(active, next_claim, claim)
        # Row rounds accumulate the bound row-eq mass into pad_adj; at the last
        # row round it becomes the scalar eq_adj and pad_adj restarts to track the
        # interaction variables' own mass (the `_run_jagged_rounds` boundary).
        eq_adj = jnp.where(active & last_row, next_pad, eq_adj)
        pad_adj = jnp.where(active, jnp.where(last_row, one, next_pad), pad_adj)
        n0, n1, d0, d1 = (
            _select_active(active, fa, pa)
            for fa, pa in ((fn0, n0), (fn1, n1), (fd0, d0), (fd1, d1))
        )
        # eq_row folds on row rounds, eq_int on interaction rounds; the inactive
        # guard subsumes the wrong-phase one (a fold there is discarded).
        eq_row = _select_active(active & in_rows, feq_row, eq_row)
        # niv == 0 has no interaction rounds: eq_int stays its natural table (the
        # empty-product eq factor), exactly as `_run_jagged_rounds` leaves it.
        # Folding a width-1 table here would collapse it to width 0 and the next
        # `eq_int[col]` gather would slice a 0-length axis.
        if niv:
            feq_int = fold_lsb(eq_int, r)
            eq_int = _select_active(active & ~in_rows, feq_int, eq_int)
        # The inactive padding rounds leave the carry untouched via the selects
        # above. Under `lax.scan` that dead-fold + select chain can alias into a
        # live earlier round and corrupt its poly (an XLA scan-body value-numbering
        # hazard, the family `zerocheck.jagged` documents). Materialize the
        # per-round carry behind an optimization barrier so each round is
        # independent. (Task 4 re-verifies whether the scan still needs it.)
        claim, eq_adj, pad_adj, n0, n1, d0, d1, eq_row, eq_int = (
            lax.optimization_barrier(
                (claim, eq_adj, pad_adj, n0, n1, d0, d1, eq_row, eq_int)
            )
        )

    # The real challenges land reversed at the tail of the fixed buffer (inactive
    # rounds are zeros at the front); roll them to the front so `chal[:real_rounds]`
    # is the bound point (a take by index -- this jaxlib lacks `jnp.roll`).
    rev = jnp.stack(challenges[::-1])
    roll = (jnp.arange(max_rounds) + (max_rounds - real_rounds)) % max_rounds
    chal = rev[roll]
    planes_final = _Planes(n0[0], n1[0], d0[0], d1[0])
    return jnp.stack(polys), chal, planes_final, transcript


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
# None under caps, so the capped pyramid shares one key). The derived schedule shrank
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
    planes = (
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    # Under caps, lay the planes into the pooled cap-width buffers BEFORE the
    # zone: the whole-layer program then keys on the cap shape -- one compile
    # per nrv class, reused across every layer, pass, AND shard -- instead of
    # on the exact per-layer/per-shard plane widths. The in-trace
    # `_pad_to_width` no-ops on an already-cap-width operand, so the zone body
    # is unchanged. Concrete path only: a tracer means an outer trace owns the
    # layout (and pooling would donate a traced value).
    if caps is not None:
        if caps.row < planes[0].shape[0]:
            raise ValueError(
                f"row cap {caps.row} cannot hold the layer's row-phase plane "
                f"width ({planes[0].shape[0]}); widen the cap (or its ladder "
                "class) so the fixed-width pad is non-negative"
            )
        # Concrete path only: a tracer means an outer trace owns the layout
        # (and pooling would donate a traced value).
        if not isinstance(planes[0], jax.core.Tracer):
            planes = tuple(
                _pool_lay_batch(
                    [
                        (role, a, caps.row)
                        for role, a in zip(("n0", "n1", "d0", "d1"), planes)
                    ]
                )
            )
    return _jagged_round_zone(
        *planes,
        _row_counts_operand(layer.row_counts),
        _round_live_meta(layer.row_counts, nrv),
        niv,
        challenge_limbs,
        caps,
        None if caps is not None else _round_out_pairs(layer.row_counts, nrv),
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

    The per-layer prove dispatches through the module-level `_jagged_round_zone`
    (one executable per layer): the schedule rides as a traced operand and the
    planes arrive already cap-width, so the whole-layer trace keys on the cap
    shape -- one compile per nrv class, reused across every layer, pass, AND
    shard. The round holds only its layer (no per-instance jit, no self-closure),
    so the chain frees each round -- and its layer -- the moment it builds the
    next. The pyramid stays a host-orchestrated Python loop of these (one trace
    per layer shape, never one `jit` over the whole pyramid -- it does not fit at
    scale; see `prover.LogupSumcheckRound`).
    """

    def __init__(
        self,
        layer: JaggedGkrLayer,
        challenge_limbs: int = 1,
        *,
        caps: RoundWidthCaps | None = None,
    ) -> None:
        # `partial` closes over (layer, challenge_limbs, caps), not `self`, so
        # the chain frees the round -- and its layer -- the moment it builds the
        # next. `caps` selects the fixed-width round layout (see
        # `prove_jagged_layer`).
        self._call = partial(_jagged_round_via_zone, layer, challenge_limbs, caps)

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


def _prove_layer(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps | None,
    carry: Carry,
    transcript: Transcript,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Prove one layer via the existing whole-layer path (operand building +
    zone). Extracted so the entry, the carve, and the scan fallback share it."""
    return _jagged_round_via_zone(layer, challenge_limbs, caps, carry, transcript)


def prove_jagged_pyramid(
    layers: list[JaggedGkrLayer],
    carry: Carry,
    transcript: Transcript,
    *,
    caps: RoundWidthCaps | None,
    challenge_limbs: int = 1,
) -> tuple[Carry, Transcript, list[JaggedLayerProof]]:
    """Prove the jagged GKR pyramid's layer chain -- byte-identical to
    ProveChain(JaggedGkrLayerRound(l) for l in layers). `layers` are the
    proved layers floor-outward, threading the `(num_eval, den_eval,
    eval_point)` carry from one layer's proof into the next's claim.

    Intent: roll the chain into a `lax.scan` when `caps` is given, bounding
    cross-layer peak memory under an outer jit (the per-layer working set
    becomes the scan carry, allocated once, reused) -- landing in a later
    task. Today this is still a per-layer Python loop; `caps=None` is a valid
    path and runs it eagerly, no caps required. Requires a device-FS
    transcript (host-FS is an eager primitive that can't trace into a scan).
    """
    layers = list(layers)
    if not layers:
        raise ValueError("prove_jagged_pyramid needs at least one layer")
    if not isinstance(transcript, DuplexTranscript):
        raise TypeError("prove_jagged_pyramid threads a DuplexTranscript scan carry")
    if transcript.fs_on_host:
        raise NotImplementedError(
            "prove_jagged_pyramid needs a device-FS transcript; host-FS is an "
            "eager primitive that cannot trace into the scan body"
        )

    # BF->EF carve: the mixed-field layer (base-field LogUp numerators under EF
    # denominators) can't ride the scan (its fold changes the carry dtype). Prove
    # it via the per-layer path; scan the all-EF interior. It is the chain's last
    # round, so the result is byte-identical.
    if layers[-1].numerator_0.dtype != carry[0].dtype:
        *interior, last = layers
        interior_proofs: list[JaggedLayerProof] = []
        if interior:
            carry, transcript, interior_proofs = prove_jagged_pyramid(
                interior, carry, transcript, caps=caps, challenge_limbs=challenge_limbs
            )
        carry, transcript, last_proof = _prove_layer(
            last, challenge_limbs, caps, carry, transcript
        )
        return carry, transcript, [*interior_proofs, last_proof]

    # TASK 1: per-layer loop (identical to ProveChain). TASK 4 replaces this with
    # the lax.scan.
    proofs = []
    for layer in layers:
        carry, transcript, proof = _prove_layer(
            layer, challenge_limbs, caps, carry, transcript
        )
        proofs.append(proof)
    return carry, transcript, proofs


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
