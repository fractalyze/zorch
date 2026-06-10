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
from jax import Array

from zorch.fusion import fused_region
from zorch.logup_gkr.circuit import JaggedGkrLayer, _pad_neutral, _segment_gather
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
    round polynomials, and the final pair openings.

    A pytree (every field is an `Array`, like the dense `sumcheck.RoundMsg`) so
    it can be returned across a `jax.jit` boundary -- the per-layer jit the
    chained prover wraps each round in."""

    lam: Array
    claim: Array
    round_polys: Array  # (num_variables, _DEGREE + 1), ascending coefficients
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
    challenges, advanced, polys, fn0, fn1, fd0, fd1 = out
    proof = JaggedLayerProof(lam, claim, polys, fn0, fn1, fd0, fd1)
    return challenges, advanced, proof


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
    permutation rides as the nested `poseidon2:` marker inside `sample_challenge`.
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

    Module-level (not a method) so `JaggedGkrLayerRound`'s optional `jax.jit`
    can close over `(layer, challenge_limbs)` via `functools.partial` without
    capturing `self` -- a `self`-closure would make the round refer to itself,
    deferring its (and its layer's) release past the chain's one-live-layer
    bound (`ChainedJaggedProveTest`)."""
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


class JaggedGkrLayerRound(Round):
    """Prove one jagged GKR layer; the chain of these (floor outward) is the
    jagged GKR prover, threading the same `(num_eval, den_eval, eval_point)`
    carry as the dense chain. `challenge_limbs` rides on the round because
    every challenge in the layer -- lam, the per-variable folds, and the
    child-selector r -- must come from the same squeeze rule.

    The shared head `prover.bind_output` works unchanged for a jagged output
    when `challenge_limbs == 1`; a consumer squeezing multi-limb challenges
    owns its binding glue.

    With `jit=True` the per-layer prove is wrapped in `jax.jit`, closing over
    the (non-pytree) layer so only `(carry, transcript)` are traced. Each round
    instance then compiles once and dispatches the cached executable on later
    calls -- a consumer reusing the round across the pyramid's warm iters pays
    the per-layer trace + composite build once, not per call. The pyramid stays
    a host-orchestrated Python loop of these (one `jit` per layer, never one
    `jit` over the whole pyramid -- it does not fit at scale; see
    `prover.LogupSumcheckRound`).
    """

    def __init__(
        self, layer: JaggedGkrLayer, challenge_limbs: int = 1, *, jit: bool = False
    ) -> None:
        # `partial` closes over the layer (and limb count), not `self`, so the
        # round holds no reference to itself -- the chain can drop it (and free
        # its layer) the moment the next round is built.
        body = partial(_prove_jagged_layer_round, layer, challenge_limbs)
        self._call = jax.jit(body) if jit else body

    def __call__(
        self, carry: Carry, transcript: Transcript
    ) -> tuple[Carry, Transcript, JaggedLayerProof]:
        return self._call(carry, transcript)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = JaggedGkrLayerRound
