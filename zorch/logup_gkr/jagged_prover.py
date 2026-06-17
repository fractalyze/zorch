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
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from zorch.fusion import fused_region
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _fixed_width_gather,
    _gather_pad,
    _pad_neutral,
    _pad_to_width,
    _segment_gather,
)
from zorch.logup_gkr.prover import Carry, logup_combine
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
)
from zorch.transcript import DuplexState, DuplexTranscript, Transcript, sample_challenge
from zorch.utils.bits import log2_ceil_usize

if TYPE_CHECKING:
    from zorch.round import ProverRound

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2), in coefficient form.
_DEGREE = 3

# Mirror of zkx's jagged on-chip budget / peel-chain envelope
# (zkx/service/transforms/sumcheck_rewriter.cc: JaggedOnChipBudget,
# JaggedFsReuseRounds, JaggedPeelChainNumVarsMax). When the static envelope
# row-count sum exceeds the budget the vendor's masked capacity-split chain
# fires, and it requires a runtime_row_counts marker's `num_vars` to be the
# fixed envelope `k_max + tail_max` (a scan-invariant chain shape), NOT the
# real round count. These constants are GPU-budget-derived but shard-invariant.
_JAGGED_FS_SHARED_CAP_BYTES = 48 * 1024  # default static shared-memory cap
_JAGGED_FS_SHARED_MULT = 12
_JAGGED_PEEL_CHAIN_MAX_PEELS = 16


def _jagged_on_chip_budget(factor_bytes: int) -> int:
    return _JAGGED_FS_SHARED_CAP_BYTES // (_JAGGED_FS_SHARED_MULT * factor_bytes)


def _jagged_peel_chain_num_vars_max(budget: int) -> int:
    if budget <= 0:
        return 0
    # JaggedFsReuseRounds = 2 * ceil(log2(budget)).
    return _JAGGED_PEEL_CHAIN_MAX_PEELS + 2 * log2_ceil_usize(budget)


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
    meta = []
    counts = row_counts
    for _ in range(num_row_vars):
        padded = tuple(rc + rc % 2 for rc in counts)
        pairs = tuple(p // 2 for p in padded)
        col_index = jnp.asarray(np.repeat(np.arange(len(pairs), dtype=np.int32), pairs))
        pair_index = jnp.asarray(
            np.concatenate([np.arange(pc, dtype=np.int32) for pc in pairs])
        )
        meta.append((_segment_gather(counts, padded), col_index, pair_index))
        counts = pairs
    return meta


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
    """Bind the LSB variable: consecutive pairs fold to `e0 + r*(e1 - e0)`."""
    return arr[0::2] + r * (arr[1::2] - arr[0::2])


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
    naturals = jnp.stack([jnp.array(j, eval_point.dtype) for j in range(_DEGREE + 1)])
    inv_vand = compute_inv_vandermonde(_DEGREE, eval_point.dtype)

    head = (n0, n1, d0, d1, eq_row, eq_int, eval_point, lam, claim)
    tail = (meta, naturals, inv_vand, nrv, niv, challenge_limbs)
    # Mark only a DuplexTranscript with a dedicated-fusion permutation, mirroring
    # `sumcheck.prover.prove`: the marker threads the sponge's leaves as operands.
    # The marked path decomposes through the same `_run_jagged_rounds`, so it is
    # byte-identical to the plain loop -- a vendor codegens it register-resident
    # over the sparse layout from the `row_counts` attribute (zkx#544). `transcript`
    # is passed positionally (not in a pre-built tuple) so the isinstance narrows it.
    if isinstance(transcript, DuplexTranscript) and transcript.has_dedicated_fusion:
        out = _prove_jagged_marked(layer, *head, transcript, *tail)
    else:
        out = _run_jagged_rounds(*head, transcript, *tail)
    bound_point, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, bound_point, fn0, fn1, fd0, fd1)
    return bound_point, advanced, proof


def _run_jagged_rounds(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_row: Array,
    eq_int: Array,
    eval_point: Array,
    lam: Array,
    claim: Array,
    transcript: Transcript,
    meta: list[tuple[Array | None, Array, Array]],
    naturals: Array,
    inv_vand: Array,
    nrv: int,
    niv: int,
    challenge_limbs: int,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-round jagged sumcheck loop, shared by the plain and marked paths so
    the `zorch.sumcheck` marker decomposes byte-identically. Returns the bound
    point (challenges reversed), the advanced transcript, the stacked round
    polynomials, and the four folded pair openings."""
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

        claim = eval_coeffs(poly, r)
        pad_adj = pad_adj * (point[-1] * r + (one - point[-1]) * (one - r))
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


def _prove_jagged_marked(
    layer: JaggedGkrLayer,
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_row: Array,
    eq_int: Array,
    eval_point: Array,
    lam: Array,
    claim: Array,
    transcript: DuplexTranscript,
    meta: list[tuple[Array | None, Array, Array]],
    naturals: Array,
    inv_vand: Array,
    nrv: int,
    niv: int,
    challenge_limbs: int,
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
            bn0,
            bn1,
            bd0,
            bd1,
            beq_row,
            beq_int,
            beval,
            blam,
            bclaim,
            DuplexTranscript(perm, rate, DuplexState(*lv)),
            bmeta,
            bnaturals,
            binv_vand,
            nrv,
            niv,
            challenge_limbs,
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
    num_eval = n0 + (n1 - n0) * r
    den_eval = d0 + (d1 - d0) * r
    # MSB-first point + the pyramid's child selector as the low (last) bit.
    eval_point = jnp.concatenate([point, jnp.atleast_1d(r)])
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


def _layer_plane_width(row_counts: tuple[int, ...], nrv: int, niv: int) -> int:
    """Fixed buffer width for one layer: the widest per-round prepad height (an
    odd segment padding up to even can exceed the layer's own height) or the
    dense interaction width, whichever is larger. The rolled pyramid pads every
    layer to the max of this across the chain so the scan stays shape-invariant."""
    width = 1 << niv
    counts = row_counts
    for _ in range(nrv):
        width = max(width, sum(rc + rc % 2 for rc in counts))
        counts = tuple((rc + rc % 2) // 2 for rc in counts)
    return width


def _padded_round_schedule(
    row_counts: tuple[int, ...],
    nrv: int,
    niv: int,
    max_rounds: int,
    plane_width: int,
) -> dict[str, np.ndarray]:
    """Host-side per-round schedule for one layer, padded to the chain's fixed
    `max_rounds` / `plane_width` so every layer's schedule stacks into one
    shape-invariant `lax.scan` xs.

    Each of `max_rounds` entries carries everything `_run_jagged_rounds`'s round
    body reads off the static layout: the segment gather re-padding the plane
    buffer (sentinels past the live rows resolve to the neutral fraction), the
    `eq_row` pair lookups and the `eq_int` per-pair interaction weights for a row
    round, the live pair count the materialized sum is masked to, and the
    `active` / `in_rows` / `last_row` flags that drive the fixed loop. Rounds
    past this layer's `nrv + niv` are INACTIVE -- a shorter floor layer must not
    advance Fiat-Shamir there -- and rounds in `[nrv, nrv + niv)` are the dense
    interaction rounds, so their gather is the identity over the live prefix and
    their pair lookups are the stride-2 `eq_int` split.

    Indices for inactive (or wrong-phase) rounds are filled with in-bounds zeros;
    the live-pair mask and the `active` flag zero their contribution out.
    """
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
        # col_index / pair_index are static host-side schedule, derived from the
        # Python-int row counts -- kept in numpy here so the schedule stacks
        # under jit (the rolled scan body reads them off the static layout).
        # _round_metadata wraps the same values in jnp for the unrolled jax
        # round body; np.asarray on that jnp array is a tracer under jit.
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


def _excl_cumsum(x: Array) -> Array:
    """Exclusive prefix sum ``out[i] = sum(x[:i])`` via a broadcast mask -- this
    jax fork has no ``jnp.cumsum`` and ``nseg`` is tiny, so the n^2 cost is nil."""
    idx = jnp.arange(x.shape[0])
    return jnp.sum(jnp.where(idx[None, :] < idx[:, None], x[None, :], 0), axis=1)


def _padded_round_schedule_jax(
    row_counts: Array,
    nrv: Array,
    niv: int,
    max_rounds: int,
    plane_width: int,
) -> dict[str, Array]:
    """`_padded_round_schedule` reconstructed from a RUNTIME ``row_counts`` (jax
    ``s32[nseg]``) and runtime ``nrv`` -- no host-baked plane-width arrays, so the
    rolled scan computes its schedule from the compact row-count channel instead
    of stacking a multi-GB constant. Byte-identical to the
    numpy ``_padded_round_schedule``.

    The row counts halve each round, and ``ceil(ceil(x/2)/2 ...)`` collapses to a
    single ceil-divide, so ``counts_k = ceil(row_counts / 2^k)``. Every index
    pattern is then ``iota`` + segment offsets (``searchsorted`` on the exclusive
    cumsum); the only constants left are scalars (``iota`` lowers to an op, not a
    `DenseElementsAttr`).
    """
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
            # `bk` nears 2^31, reachable when the marked split path widens
            # max_rounds to the peel-chain envelope.
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


def _run_jagged_rounds_padded(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_row: Array,
    eq_int: Array,
    coords: Array,
    lam: Array,
    claim: Array,
    transcript: Transcript,
    sched: dict[str, Array],
    naturals: Array,
    inv_vand: Array,
    niv: int,
    max_rounds: int,
    challenge_limbs: int,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """Fixed-width sibling of `_run_jagged_rounds`: run a fixed `max_rounds` loop
    over neutral-padded buffers so a whole pyramid of differently-sized layers
    shares one traced round body (the chain scans this once per layer).

    The four planes ride a fixed `plane_width` buffer with the live prefix at the
    front and the neutral fraction (n=0, d=1) in the tail; the per-round `sched`
    re-pads and folds inside that width, masking the materialized sum to the live
    pairs so the dead tail never enters it (field zero-adds are exact, so this is
    byte-equal to `_run_jagged_rounds`'s live-only sum). `coords` is the bound
    point's coordinates in round order (`eval_point` reversed), one per round.

    INACTIVE rounds (`sched["active"]` false -- a shorter layer's padding rounds
    past `nrv + niv`) select every carried value unchanged, the transcript's five
    sponge leaves included, so the layer never over-advances Fiat-Shamir; the
    chain is k separate sumchecks, not one joint scan, so this guard is what the
    homogeneous `zorch.sumcheck` scan does not need."""
    one = jnp.ones((), coords.dtype)
    zero = jnp.zeros((), coords.dtype)
    eq_adj = one
    pad_adj = one
    half_width = n0.shape[0] // 2
    pair_axis = jnp.arange(half_width)
    polys: list[Array] = []
    challenges: list[Array] = []

    for rnd in range(max_rounds):
        active = sched["active"][rnd]
        in_rows = sched["in_rows"][rnd]
        last_row = sched["last_row"][rnd]
        z_cur = coords[rnd]
        gather = sched["gather"][rnd]
        live_mask = (pair_axis < sched["live_pairs"][rnd]) & active

        pn0, pn1, pd0, pd1 = (
            _gather_pad(n0, gather, 0),
            _gather_pad(n1, gather, 0),
            _gather_pad(d0, gather, 1),
            _gather_pad(d1, gather, 1),
        )
        # Row rounds weight each pair by its segment-local row-eq factor times its
        # interaction's eq_int column; interaction rounds use the dense stride-2
        # eq_int split. Both selections are in-bounds (sentinel-free), so the
        # wrong-phase one is harmless -- the live mask zeroes it out anyway.
        # Both phases' weights are computed each round and `in_rows` selects one;
        # the discarded phase's pair indices can fall outside its (smaller) eq
        # table (a row round's `pair_lo` overruns `eq_int`, an interaction round's
        # overruns `eq_row` when niv exceeds the row width). JAX clamps an
        # out-of-bounds gather, but clamp explicitly so the selected value's
        # in-bounds invariant doesn't ride on that and the dead branch stays safe.
        w = eq_int[sched["col"][rnd]]
        lo, hi = sched["pair_lo"][rnd], sched["pair_hi"][rnd]
        eq0_row = eq_row[jnp.minimum(lo, eq_row.shape[0] - 1)] * w
        eq1_row = eq_row[jnp.minimum(hi, eq_row.shape[0] - 1)] * w
        eq0_int = eq_int[jnp.minimum(lo, eq_int.shape[0] - 1)]
        eq1_int = eq_int[jnp.minimum(hi, eq_int.shape[0] - 1)]
        eq0 = jnp.where(in_rows, eq0_row, eq0_int)
        eq1 = jnp.where(in_rows, eq1_row, eq1_int)
        eq0 = jnp.where(live_mask, eq0, jnp.zeros((), eq0.dtype))
        eq1 = jnp.where(live_mask, eq1, jnp.zeros((), eq1.dtype))

        eval_zero, eval_half, eq_sum = _paired_sums(pn0, pn1, pd0, pd1, eq0, eq1, lam)
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

        observed = transcript.observe(poly)
        observed, r = sample_challenge(observed, claim.dtype, challenge_limbs)
        # Inactive rounds leave the sponge (all five leaves) and the sampled
        # challenge untouched; the proof slot they emit is sliced off host-side.
        transcript = _select_transcript(active, observed, transcript)
        r = jnp.where(active, r, zero)
        polys.append(jnp.where(active, poly, jnp.zeros_like(poly)))
        challenges.append(r)

        next_claim = eval_coeffs(poly, r)
        next_pad = pad_adj * (z_cur * r + (one - z_cur) * (one - r))
        fn0, fn1, fd0, fd1 = (_bind_lsb(a, r) for a in (pn0, pn1, pd0, pd1))
        feq_row = _bind_lsb(eq_row, r)
        feq_int = _bind_lsb(eq_int, r)

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
        eq_int = _select_active(active & ~in_rows, feq_int, eq_int)
        # The inactive padding rounds (a shorter layer's tail) leave the carry
        # untouched via the selects above. Under `lax.scan` that dead-fold +
        # select chain can alias into a live earlier round and corrupt its poly
        # (an XLA scan-body value-numbering hazard; the same family
        # `zerocheck.jagged` documents). Materialize the per-round carry behind an
        # optimization barrier so each round's state is independent.
        claim, eq_adj, pad_adj, n0, n1, d0, d1, eq_row, eq_int = (
            lax.optimization_barrier(
                (claim, eq_adj, pad_adj, n0, n1, d0, d1, eq_row, eq_int)
            )
        )

    return (
        jnp.stack(challenges[::-1]),
        transcript,
        jnp.stack(polys),
        n0[0],
        n1[0],
        d0[0],
        d1[0],
    )


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
    need. Both are `DuplexTranscript`s built off the same permutation/rate."""
    a = cast(DuplexTranscript, advanced)
    k = cast(DuplexTranscript, kept)
    sel = lambda x, y: jnp.where(active, x, y)
    return DuplexTranscript(
        a.permutation,
        a.rate,
        DuplexState(
            sel(a.state.input_buffer, k.state.input_buffer),
            sel(a.state.output_buffer, k.state.output_buffer),
            sel(a.state.sponge_state, k.state.sponge_state),
            sel(a.state.in_pos, k.state.in_pos),
            sel(a.state.out_pos, k.state.out_pos),
        ),
    )


# The rolled scan step's per-round schedule, threaded through the marker as
# operands (the body rebuilds the dict from them in this fixed order). Shared by
# the marker producer and `prove_jagged_pyramid`'s xs stacking so they agree.
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


def _prove_jagged_rounds_padded_marked(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq_row: Array,
    eq_int: Array,
    coords: Array,
    lam: Array,
    claim: Array,
    transcript: Transcript,
    sched: dict[str, Array],
    naturals: Array,
    inv_vand: Array,
    bound_meta: Array,
    row_counts: Array,
    *,
    envelope: tuple[int, ...],
    niv: int,
    max_rounds: int,
    challenge_limbs: int,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """Wrap the rolled fixed-width round loop in the `zorch.sumcheck` composite,
    Fiat-Shamir INSIDE, so the body is the same `_run_jagged_rounds_padded` and the
    decomposition is bit-identical to the plain padded path.

    The pyramid scan traces ONE body for every layer, so a single static
    `row_counts` attribute cannot carry the per-layer counts (the pyramid halves
    each layer). Dual-channel instead: the static `row_counts` attr
    pins the scan-invariant fixed-width ENVELOPE (per-segment max across layers),
    `runtime_row_counts` flags the channel, and the per-layer actual counts ride as
    the trailing `bound_meta s32[4]={num_vars, nrv, nseg, num_blocks}` /
    `row_counts s32[nseg]` runtime operands the zkx recognizer/rewriter consume
    (sumcheck_recognizer.cc). Both are rank-1 s32, so the rewriter's duplex-leaf
    scan -- which keys on the rank-0 s32 in_pos/out_pos -- never mistakes them for a
    transcript leaf. On CPU the marker is unrecognized and decomposes through the
    same loop, so the runtime operands are dead there: the schedule operands
    already carry the per-layer layout. The caller's `mark` gate guarantees a
    dedicated-fusion `DuplexTranscript` here (the plain path takes a cheap one)."""
    duplex = cast(DuplexTranscript, transcript)
    perm, rate = duplex.permutation, duplex.rate
    leaves = _state_leaves(duplex.state)
    sched_ops = [sched[k] for k in _SCHED_KEYS]

    # The GPU emitter reads `z_cur = point[num_vars - 1 - rnd]` over the per-layer
    # ACTUAL num_vars (from bound_meta), so operand-6 must be the MSB-first
    # eval_point with the live coords at the FRONT ([0, num_vars)). This gather
    # converts between round order and that eval_point order (an involution on the
    # live prefix): a plain `[::-1]` reverses over the padded `max_rounds`, which
    # only matches when num_vars == max_rounds (the largest / single layer) and
    # otherwise strands a shorter pyramid layer's live coords at the padded tail,
    # making the emitter read the neutral pad.
    def _swap_point_order(arr: Array, num_vars: Array) -> Array:
        return arr[jnp.clip(num_vars - 1 - jnp.arange(max_rounds), 0, max_rounds - 1)]

    def body(*operands: Array, **_attrs: object) -> tuple[Array, ...]:
        bn0, bn1, bd0, bd1, beq_row, beq_int, bpoint, blam, bclaim = operands[:9]
        idx = 9
        lv = operands[idx : idx + len(leaves)]
        idx += len(leaves)
        bsched = dict(zip(_SCHED_KEYS, operands[idx : idx + len(_SCHED_KEYS)]))
        idx += len(_SCHED_KEYS)
        bnaturals, binv_vand = operands[idx], operands[idx + 1]
        # operands[idx + 2 :] are (bound_meta, row_counts): the GPU recognizer's
        # runtime row-count channel, dead on the CPU decomposition below. Index
        # positively (not operands[-2]) -- a traced body may get auto-lifted
        # constants appended to the operand tail. bound_meta = {num_vars, nrv,
        # nseg, num_blocks}.
        bbound_meta = operands[idx + 2]
        bcoords = _swap_point_order(bpoint, bbound_meta[0])
        challenges, t, polys, fn0, fn1, fd0, fd1 = _run_jagged_rounds_padded(
            bn0,
            bn1,
            bd0,
            bd1,
            beq_row,
            beq_int,
            bcoords,
            blam,
            bclaim,
            DuplexTranscript(perm, rate, DuplexState(*lv)),
            bsched,
            bnaturals,
            binv_vand,
            niv,
            max_rounds,
            challenge_limbs,
        )
        # Result order is the recognizer's contract, shared with the dense and
        # unrolled jagged markers: [folded][5 sponge leaves][round polys][challenges].
        leaves_out = _state_leaves(cast(DuplexTranscript, t).state)
        return (fn0, fn1, fd0, fd1, *leaves_out, polys, challenges)

    eval_point_op = _swap_point_order(coords, bound_meta[0])

    out = fused_region(
        body,
        n0,
        n1,
        d0,
        d1,
        eq_row,
        eq_int,
        eval_point_op,
        lam,
        claim,
        *leaves,
        *sched_ops,
        naturals,
        inv_vand,
        bound_meta,
        row_counts,
        name=SUMCHECK_MARKER,
        version=SUMCHECK_MARKER_VERSION,
        degree=_DEGREE,
        num_vars=max_rounds,
        num_factors=4,
        row_counts=np.asarray(envelope, dtype=np.int64),
        runtime_row_counts=True,
        fold_order="lsb",
        poly_form="coefficient",
    )
    fn0, fn1, fd0, fd1, *out_leaves, polys, challenges = out
    t = DuplexTranscript(perm, rate, DuplexState(*out_leaves))
    return challenges, t, polys, fn0, fn1, fd0, fd1


def prove_jagged_pyramid(
    layers: Sequence[JaggedGkrLayer],
    carry: Carry,
    transcript: Transcript,
    *,
    challenge_limbs: int = 1,
) -> tuple[Carry, Transcript, list[JaggedLayerProof]]:
    """Prove the jagged GKR pyramid as ONE `lax.scan` over the floor-outward
    layer chain -- O(1) in the layer count, byte-identical to the unrolled
    `ProveChain(JaggedGkrLayerRound(l) for l in layers)`.

    `layers` are the proved layers floor-outward (the chain's `reversed(
    layers[:-1])`); the pyramid halves each layer, so the per-layer planes ride
    fixed-width buffers padded to the first (largest) layer's height with the
    live prefix at the front (the `sumcheck.prove` / `zerocheck.jagged` pattern),
    and the per-layer round count -- which grows by one each outward layer --
    rides a fixed `max_rounds` loop whose tail rounds a shorter layer leaves
    INACTIVE, selecting the unchanged transcript so it never over-advances
    Fiat-Shamir. Returns the same `(carry, transcript, per-layer proofs)` triple
    as the chain.
    """
    layers = list(layers)
    if not layers:
        raise ValueError("prove_jagged_pyramid needs at least one layer")
    if not isinstance(transcript, DuplexTranscript):
        raise TypeError("prove_jagged_pyramid threads a DuplexTranscript scan carry")

    niv = layers[0].num_interaction_variables
    num_eval0, _den0, eval_point0 = carry
    dtype = num_eval0.dtype
    # nrv grows by one each outward layer (the carry's eval_point gains the child
    # selector), so the schedule's per-layer round count is host-known from the
    # entry eval_point length; the planes' fixed width is the first (largest)
    # layer's padded round-0 height.
    init_nrv = int(eval_point0.shape[0]) - niv
    nrvs = [init_nrv + j for j in range(len(layers))]
    if any(layer.num_interaction_variables != niv for layer in layers):
        raise ValueError("every layer must share the interaction-variable count")
    real_rounds = [nrv + niv for nrv in nrvs]
    max_rounds = max(real_rounds)
    # The carry's eval_point grows by one per layer (the child selector), so the
    # exit point of the last layer is the widest -- the fixed scan-carry width.
    max_eval_len = max_rounds + 1

    # The scan-invariant buffer width: the max per-layer width across the chain.
    plane_width = max(
        _layer_plane_width(layer.row_counts, nrv, niv)
        for layer, nrv in zip(layers, nrvs)
    )

    # Dual-channel row_counts for the marked path: nseg is constant
    # across the pyramid (a layer transition preserves the interaction count), so
    # the runtime `row_counts s32[nseg]` operand stays shape-invariant. The static
    # marker attribute pins the scan-invariant fixed-width envelope (per-segment
    # max -- the counts grow toward it floor-outward), each layer's actual counts
    # ride the runtime operand. A dedicated-fusion transcript marks; a cheap one
    # decomposes inline, mirroring the unrolled `prove_jagged_layer` gate.
    nseg = len(layers[0].row_counts)
    if any(len(layer.row_counts) != nseg for layer in layers):
        raise ValueError("every layer must share the segment (interaction) count")
    envelope = tuple(max(layer.row_counts[s] for layer in layers) for s in range(nseg))
    mark = transcript.has_dedicated_fusion

    # A split-needing marked composite must size num_vars to zkx's fixed
    # peel-chain envelope (k_max + tail_max), NOT the real round count, so the
    # vendor's runtime-masked split chain keeps a scan-invariant shape. The
    # surplus rounds run inactive (neutral), so each layer's proof, sliced to
    # its real round count, stays byte-identical.
    if mark:
        budget = _jagged_on_chip_budget(int(np.dtype(dtype).itemsize))
        if budget > 0 and sum(envelope) > budget:
            max_rounds = max(max_rounds, _jagged_peel_chain_num_vars_max(budget))
            max_eval_len = max_rounds + 1

    # eq_row only needs the natural row hypercube. Each layer folds eq_row exactly
    # its `nrv` (= eval_len - niv <= max(nrvs)) row rounds, and the LSB fold's
    # one-past read `eq_row[1]` at the last row round reaches eq[2^(nrv-1):2^nrv],
    # so the consumed extent is the full 2^(max row vars). When a split layer bumps
    # `max_rounds` to the peel-chain envelope above, the surplus rounds run inactive
    # and never fold eq_row, so sizing the build to the bumped `1 << (max_rounds -
    # niv)` would materialize a table whose tail is identity-replicated and never
    # read into the live region. Build only the natural extent (the trim is
    # byte-exact: a smaller prefix of the same eq table, folded identically).
    row_var_extent = max(nrvs)
    eq_prefix_width = 1 << row_var_extent

    one = jnp.ones((), dtype)
    zero = jnp.zeros((), dtype)
    naturals = jnp.stack([jnp.array(j, dtype) for j in range(_DEGREE + 1)])
    inv_vand = compute_inv_vandermonde(_DEGREE, dtype)

    # The bound point's live length per layer (the entry eval_point length), used
    # to read each round's coordinate and to slice the new eval_point back.
    xs_eval_len = jnp.asarray(np.asarray(real_rounds, dtype=np.int32))
    # Per-layer runtime row-count channel: drives the in-body schedule
    # reconstruction (`_padded_round_schedule_jax`, #109 -- no baked plane-width
    # schedule) AND feeds the marker's runtime row-count operand. `bound_meta`
    # {num_vars, nrv, nseg, num_blocks} is the per-layer metadata the recognizer
    # reads instead of the static envelope. num_blocks=1 is a placeholder a split
    # layer's rewriter recomputes from its grid (zkx#641); the rolled marker
    # always carries both so the scan body stays shape-invariant.
    xs_bound_meta = jnp.asarray(
        np.asarray(
            [[rounds, nrv, nseg, 1] for rounds, nrv in zip(real_rounds, nrvs)],
            dtype=np.int32,
        )
    )
    xs_row_counts = jnp.asarray(
        np.asarray([layer.row_counts for layer in layers], dtype=np.int32)
    )

    # Each layer's interaction-major planes ride a flat per-channel buffer at
    # their NATURAL width (concatenated): a halving pyramid's natural widths sum
    # to ~2 · plane_width, so peak plane residency stays O(plane_width),
    # independent of the layer count. `step` slices each layer's `plane_width`
    # window out of the flat buffer and masks the tail past the live width back
    # to the neutral fraction, so the round body sees the [live prefix | neutral
    # tail] plane (byte-identical to `_pad_to_width`).
    # Per-layer natural widths (`JaggedGkrLayer.__post_init__` guarantees all four
    # channels are flat over `height`) and their exclusive-prefix offsets into the
    # flat buffer.
    plane_widths = [layer.height for layer in layers]
    offsets = np.concatenate([[0], np.cumsum(plane_widths)[:-1]]).astype(np.int32)
    # A `plane_width` window at the last (widest-offset) layer must stay in bounds
    # -- `dynamic_slice` clamps a past-the-end start, which would read the wrong
    # layer -- so size the buffer to the last window's end (>= the natural sum,
    # since `plane_width` is the per-layer max width).
    flat_len = int(offsets[-1]) + plane_width

    def _flat_channel(attr: str, neutral: int) -> Array:
        flat = jnp.concatenate([getattr(layer, attr).astype(dtype) for layer in layers])
        return _pad_to_width(flat, flat_len, neutral)

    flat_n0 = _flat_channel("numerator_0", 0)
    flat_n1 = _flat_channel("numerator_1", 0)
    flat_d0 = _flat_channel("denominator_0", 1)
    flat_d1 = _flat_channel("denominator_1", 1)
    xs_offsets = jnp.asarray(offsets)
    xs_widths = jnp.asarray(np.asarray(plane_widths, dtype=np.int32))

    perm, rate = transcript.permutation, transcript.rate

    def step(
        carry_scan: tuple[Carry, Transcript], xs: tuple
    ) -> tuple[tuple[Carry, Transcript], tuple]:
        (num_eval, den_eval, eval_point), t = carry_scan
        offset, live_width, eval_len, row_counts, bound_meta = xs

        # Reconstruct this layer's fixed-width planes from the flat per-channel
        # buffers: a `plane_width` window at the layer's offset, with the tail
        # past the live width masked to the neutral fraction (0 numerator, 1
        # denominator) -- the [live prefix | neutral tail] the round body expects.
        live = jnp.arange(plane_width) < live_width

        def window(flat: Array, neutral: Array) -> Array:
            win = lax.dynamic_slice_in_dim(flat, offset, plane_width, 0)
            return jnp.where(live, win, neutral)

        n0, n1 = window(flat_n0, zero), window(flat_n1, zero)
        d0, d1 = window(flat_d0, one), window(flat_d1, one)

        # Per-layer carry reduction head: sample lam, batch the opening claim.
        t, lam = sample_challenge(t, dtype, challenge_limbs)
        claim = lam * num_eval + den_eval

        # eq tables over the entry point: rows are the trailing `nrv` coords,
        # interactions the leading `niv` -- nrv is host-known per layer, so the
        # split index rides as a traced eval_len. Build at the fixed widths.
        nrv_t = eval_len - niv

        # Reconstruct this layer's per-round schedule from the compact row counts
        # (no baked plane-width constant); byte-identical to the host-baked
        # `_padded_round_schedule`.
        sched = _padded_round_schedule_jax(
            row_counts, nrv_t, niv, max_rounds, plane_width
        )
        # eq_row spans 2^nrv live entries; expand over the natural-extent row coords
        # into the trimmed buffer (see row_var_extent / eq_prefix_width above). Both
        # are the pre-envelope-bump natural extent, so the surplus split rounds never
        # enter the build and the schedule's pair lookups stay within the live region.
        row_pt = lax.dynamic_slice_in_dim(eval_point, niv, row_var_extent, 0)
        eq_row = _expand_eq_prefix(row_pt, nrv_t, eq_prefix_width, one)
        # eq_int stays its natural 2^niv width: it folds via `_bind_lsb` (needs an
        # even length) and the schedule's lookups never index past 2^niv - 1.
        eq_int = expand_eq_to_hypercube(eval_point[:niv], one)

        # coords[rnd] = eval_point[eval_len - 1 - rnd]: the round consumes the
        # point from the end, and the bound point is the challenges reversed.
        idx = eval_len - 1 - jnp.arange(max_rounds)
        coords = eval_point[jnp.clip(idx, 0, eval_point.shape[0] - 1)]

        # A dedicated-fusion transcript wraps the round loop in the `zorch.sumcheck`
        # marker (a vendor codegens it register-resident over the runtime row
        # counts); a cheap one runs the plain loop. The marker decomposes to the
        # same loop, so both are byte-identical -- the `mark` branch is host-static,
        # so the scan still traces one body. Both drivers share these positional
        # args; the marked path also threads the runtime (bound_meta, row_counts)
        # operands and the static envelope.
        round_args = (
            n0,
            n1,
            d0,
            d1,
            eq_row,
            eq_int,
            coords,
            lam,
            claim,
            t,
            sched,
            naturals,
            inv_vand,
        )
        if mark:
            challenges, t, polys, fn0, fn1, fd0, fd1 = (
                _prove_jagged_rounds_padded_marked(
                    *round_args,
                    bound_meta,
                    row_counts,
                    envelope=envelope,
                    niv=niv,
                    max_rounds=max_rounds,
                    challenge_limbs=challenge_limbs,
                )
            )
        else:
            challenges, t, polys, fn0, fn1, fd0, fd1 = _run_jagged_rounds_padded(
                *round_args, niv, max_rounds, challenge_limbs
            )

        t = t.observe(jnp.stack([fn0, fn1, fd0, fd1]))
        t, r = sample_challenge(t, dtype, challenge_limbs)
        num_eval = fn0 + (fn1 - fn0) * r
        den_eval = fd0 + (fd1 - fd0) * r
        # New eval_point = [bound point (the layer's real reversed challenges),
        # child selector r]; the live length grows by one. `_run_jagged_rounds_
        # padded` returns the full-width challenges reversed, so the real ones sit
        # at the tail (the inactive rounds are zeros at the front); roll them to
        # the front (a take by index, this jaxlib lacks `jnp.roll`), then append r
        # as the low (last) bit.
        roll = (jnp.arange(max_rounds) + (max_rounds - eval_len)) % max_rounds
        bound = jnp.concatenate([challenges[roll], jnp.zeros((1,), dtype)])
        new_point = lax.dynamic_update_index_in_dim(bound, r, eval_len, 0)

        out = (lam, claim, polys, challenges, fn0, fn1, fd0, fd1)
        return ((num_eval, den_eval, new_point), t), out

    # eval_point rides a fixed max-width buffer with the live prefix at front.
    eval_buf = jnp.concatenate(
        [eval_point0, jnp.zeros((max_eval_len - eval_point0.shape[0],), dtype)]
    )
    init = ((num_eval0, carry[1], eval_buf), transcript)
    xs = (xs_offsets, xs_widths, xs_eval_len, xs_row_counts, xs_bound_meta)
    (final_carry, final_t), outs = lax.scan(step, init, xs)
    lam_s, claim_s, polys_s, chal_s, fn0_s, fn1_s, fd0_s, fd1_s = outs

    # Reconstruct the ragged per-layer proofs: slice each layer's padded polys /
    # bound point to its real round count (the challenges land reversed, at the
    # tail of the fixed buffer, so the live point is the last `rounds` entries).
    proofs: list[JaggedLayerProof] = []
    for j, rounds in enumerate(real_rounds):
        polys_j = polys_s[j][:rounds]
        point_j = chal_s[j][max_rounds - rounds :]
        proofs.append(
            JaggedLayerProof(
                lam_s[j],
                claim_s[j],
                polys_j,
                point_j,
                fn0_s[j],
                fn1_s[j],
                fd0_s[j],
                fd1_s[j],
            )
        )

    (num_eval, den_eval, eval_buf) = final_carry
    final_eval = eval_buf[: real_rounds[-1] + 1]
    return (num_eval, den_eval, final_eval), final_t, proofs


def _expand_eq_prefix(
    point: Array, live_len: Array, width: int, scalar: Array
) -> Array:
    """`expand_eq_to_hypercube` over a traced-length prefix of `point`, into a
    fixed `width` buffer. Each doubling step is truncated to `width`, so the result
    is exactly the first `width` entries of the `2^live_len` eq table -- the prefix
    is closed under the interleave (out[0:width] depends only on
    prev[0:ceil(width/2)]). Coordinates past `live_len` fold in as the identity
    (their factor is 1). When `width >= 2^live_len` the live values fill and the
    surplus replicates harmlessly; when `width < 2^live_len` (the trimmed regime)
    only the consumed prefix is built. Either way the schedule's pair lookups and
    the LSB fold only ever touch this region. `width` is a single host-static value
    (the max consumed extent across layers) so the buffer is shape-invariant."""
    n = point.shape[0]
    state = jnp.atleast_1d(scalar)
    state = jnp.concatenate([state, jnp.zeros((width - 1,), state.dtype)])
    for j in range(n):
        coord = jnp.where(j < live_len, point[j], jnp.zeros((), point.dtype))
        # expand only the live prefix; build result[2i]/[2i+1] in place.
        low = state * (jnp.ones((), point.dtype) - coord)
        high = state * coord
        inter = jnp.stack([low, high], axis=-1).reshape(-1)[:width]
        do_expand = j < live_len
        state = jnp.where(do_expand, inter, state)
    return state


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
