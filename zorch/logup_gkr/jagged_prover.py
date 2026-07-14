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
homogeneous `zorch.sumcheck` scan (see docs/reference/conventions.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_fix_last,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_rounds import _round_interp_constants
from zorch.logup_gkr._jagged_types import _JaggedState, _Planes, _RoundScalars
from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.logup_gkr.prover import Carry, fold_carry
from zorch.round import Round
from zorch.sumcheck.jagged.buffers import (
    _pad_to_width,
    _pool_lay_batch,
    _resize_zero,
)
from zorch.sumcheck.jagged.fs import _fs_reduce
from zorch.sumcheck.jagged.rounds import _expand_eq_slice
from zorch.sumcheck.jagged.schedule import (
    _check_row_space,
    _dense_live_operand,
    _round_live_meta,
    _round_out_pairs,
    _row_counts_operand,
)
from zorch.sumcheck.jagged.types import (
    RoundWidthCaps,
    _InterpConsts,
    _JaggedSchedule,
)
from zorch.transcript import (
    DuplexTranscript,
    Transcript,
    reinterpret_challenge,
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
def _finalize_layer(
    planes: _Planes,
    alpha: Array,
    chal: list[Array],
    poly: list[Array],
) -> tuple[Array, Array, Array, Array, Array, Array]:
    head = _Planes(*(a[:2] for a in (planes.n0, planes.n1, planes.d0, planes.d1)))
    fn0, fn1, fd0, fd1 = _composite_fix_last(head, alpha)
    return fn0, fn1, fd0, fd1, jnp.stack(chal[::-1]), jnp.stack(poly)


def _run_jagged_rounds(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
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
        if caps.elements % 4:
            raise ValueError(
                f"elements cap {caps.elements} must be a multiple of 4 (the boundary "
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
                _pad_to_width(a, caps.elements, 0)
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
    for rnd in range(nrv + niv):
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
            # -- the live 2^niv prefix always fits in elements // 2 (the last
            # row layout, 2^(niv+1) slots, fits in the elements cap). `eq_int`
            # itself
            # rides through the handoff unchanged, at its own width, for the
            # interaction rounds below.
            if caps is None:
                eq_boundary = eq_int
            else:
                eq_boundary = _resize_zero(eq_int, caps.elements // 2)
            poly, planes, _ = fix_boundary(
                planes, eq_boundary, prev_r, scalars, consts, live
            )
            if caps is not None:
                # The handoff halves [elements] -> [elements // 2]; the dense
                # phase runs
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
        # Device FS hop + reduce -- traced into the whole-layer jit (one fused
        # region per round). Slices the next z_cur via the decremented `pos`,
        # riding the fold's dispatch instead of a standalone gather.
        transcript, r, claim, pad_adj, z_cur, pos = _fs_reduce(
            poly, transcript, pad_adj, z_cur, eval_point, pos, challenge_limbs, dtype
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
        if caps.elements < planes[0].shape[0]:
            raise ValueError(
                f"elements cap {caps.elements} cannot hold the layer's "
                f"row-phase plane width ({planes[0].shape[0]}); widen the cap "
                "(or its ladder class) so the fixed-width pad is non-negative"
            )
        # Concrete path only: a tracer means an outer trace owns the layout
        # (and pooling would donate a traced value).
        if not isinstance(planes[0], jax.core.Tracer):
            planes = tuple(
                _pool_lay_batch(
                    [
                        (role, a, caps.elements)
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


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
