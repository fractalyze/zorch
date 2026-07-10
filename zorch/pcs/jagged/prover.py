# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule jagged evaluation-proof sumcheck as a composable ``Round``.

``JaggedEvalRound`` is the stage-5 evaluation proof in zorch's IOP-``Round`` form,
so it sequences with the other stages —
``ProveChain([Commit(), LogUpGkr(), ZeroCheck(), JaggedEval()])``. It reproves
SP1's jagged PCS opening sumchecks byte-identically: the OUTER Hadamard sumcheck
``Σ_i D(i)·J̃(i)`` over the committed dense buffer (round polys + ``dense_eval``)
whose folded point feeds the INNER branching-program sumcheck reproving
``J̃(z_row, z_col, z_final)``. The stacked BaseFold open of ``D`` at ``z_final``
is the remaining half of stage 5.

SP1 folds **LSB-first** (even/odd pairing ``[0::2]``/``[1::2]``), round polys
travel in **coefficient** form ``[c0, c1, c2]``, and the proof point is the
challenge list reversed (insert-at-front). zorch's ``SumcheckRound`` / ``prove``
fold MSB-first over a fixed dense shape — they can't byte-match SP1's LSB-first
jagged schedule, so this ``Round`` runs its own loop over zorch's order-free leaf
blocks (``build_jagged_layout`` / ``bp_eval_core`` / ``eval_coeffs``), same as
``zerocheck/jagged.py``.

The inner challenges are sampled from the threaded transcript; ``z_col`` /
``z_trace`` arrive on the carry (fixed upstream — ``z_col`` at commitment,
``z_trace`` by the outer sumcheck).

References (same SP1 commit as ``zerocheck/jagged.py``):
- coefficient-form deg-2 round poly — ``process_univariate_polynomial``.
- LSB-first elimination — ``fix_last_variable_kernel`` (``dim-1-round``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import efinfo

from zorch.pcs.jagged.branching_program import _TRANSITION_ROWS, bp_eval_core
from zorch.pcs.jagged.poly import (
    _offset_bit_tensor,
    build_jagged_layout,
    build_prefix_sums,
    msb_first_bits,
    partial_eval_core,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round
from zorch.transcript import Transcript, reinterpret_challenge, sample_challenge
from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class JaggedEvalInputs:
    """Carry into ``JaggedEvalRound``: the committed columns' jagged layout plus
    the points the upstream rounds fixed.

    ``col_heights`` is the per-unit-column height list and ``all_claims`` the
    matching ``(L,)`` per-column GKR openings (see ``assemble_columns``).
    ``dense`` is the combined committed dense buffer ``D`` (both rounds' raw
    packed columns concatenated, padded to ``2^n``) over which the outer
    Hadamard sumcheck runs; the outer point ``z_final`` it produces feeds the
    inner sumcheck, so it is no longer carried in."""

    col_heights: tuple[int, ...]
    all_claims: Array
    z_row: Array
    z_col: Array
    dense: Array
    # Column-count cap (size-invariance): when set, every L-indexed
    # operand is padded to `cap_l` with fold-neutral empty-range columns so the
    # whole-layer eval zone keys on the cap, not the true column count -- one
    # compile per cap class serves every shard. None runs the exact layout.
    cap_l: int | None = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "outer_sumcheck_claim",
        "outer_sumcheck_polys",
        "outer_sumcheck_point",
        "dense_eval",
        "inner_sumcheck_polys",
        "inner_point",
        "inner_claimed_sum",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class JaggedEvalMsg:
    """Proof message: the outer Hadamard sumcheck (initial column claim, its
    coefficient-form round polys, the folded point ``z_final``, and
    ``dense_eval = D(z_final)``) and the inner branching-program sumcheck
    transcript (coefficient-form round polys, the folded point, the reproved
    claim).

    A registered pytree so it crosses the ``eval_round_core`` ``@jax.jit`` /
    ``jax.export`` boundary (mirrors ``open.py``'s ``StackedOpenProof``)."""

    outer_sumcheck_claim: Array
    outer_sumcheck_polys: Array
    outer_sumcheck_point: Array
    dense_eval: Array
    inner_sumcheck_polys: Array
    inner_point: Array
    inner_claimed_sum: Array


def assemble_columns(
    row_counts_rounds: Sequence[Sequence[int]],
    column_counts_rounds: Sequence[Sequence[int]],
    column_claims_rounds: Sequence[Array],
    *,
    dtype: Any,
) -> tuple[list[int], Array]:
    """Flatten the per-round (row_counts, column_counts, real claims) into the
    per-unit-column height list and the full column-claim buffer.

    Each chip contributes ``column_count`` unit columns of height ``row_count``;
    the last two ``column_counts`` per round are SP1's stacking dummies, so the
    claim buffer appends ``cc[-2]+cc[-1]`` zero claims after each round's real
    ones (matching SP1's ``prove_trusted_evaluations`` layout)."""
    col_heights: list[int] = []
    claim_blocks: list[Array] = []
    for rcs, ccs, claims_r in zip(
        row_counts_rounds, column_counts_rounds, column_claims_rounds, strict=True
    ):
        for rc, cc in zip(rcs, ccs, strict=True):
            col_heights.extend([int(rc)] * int(cc))
        if len(ccs) < 2:
            raise ValueError(
                f"each round needs the trailing (stacking-dummy, leftover) "
                f"column-count pair; got {len(ccs)} counts"
            )
        n_pad = int(ccs[-2]) + int(ccs[-1])
        claim_blocks.append(jnp.asarray(claims_r, dtype=dtype))
        if n_pad:
            claim_blocks.append(jnp.zeros((n_pad,), dtype=dtype))
    return col_heights, jnp.concatenate(claim_blocks, axis=0)


def sample_z_col(
    transcript: Transcript, num_columns: int, dtype: Any
) -> tuple[Transcript, Array]:
    """One extension challenge per column variable — SP1 samples ``z_col`` as
    extension elements, not stacked base squeezes. One definition driven by
    the prover stage and its verifier dual."""
    limbs = efinfo(dtype).degree
    parts: list[Array] = []
    for _ in range(log2_ceil_usize(num_columns)):
        transcript, challenge = sample_challenge(transcript, dtype, limbs)
        parts.append(challenge)
    z_col = jnp.stack(parts) if parts else jnp.zeros((0,), dtype)
    return transcript, z_col


def merged_prefix_bits(
    col_heights: Sequence[int], num_bits: int, *, dtype: Any
) -> Array:
    """The ``(L, 2·num_bits)`` merged prefix-bit buffer ``bits(t_c) ‖
    bits(t_{c+1})`` — the branching-program input both the inner sumcheck and
    its verifier leaf check read."""
    prefix_int = build_prefix_sums(list(col_heights))
    bits = msb_first_bits(prefix_int, num_bits)
    return jnp.asarray(np.concatenate([bits[:-1], bits[1:]], axis=1), dtype=dtype)


def outer_sumcheck_claim(all_claims: Array, z_col: Array) -> Array:
    """``Σ_c eq(z_col, c)·claim[c]`` over the real columns of the 2^⌈log L⌉ hypercube.

    The eq tail past ``L`` would multiply zero-padded claims, so summing only the
    real columns (``col_eq[:L]``) is identical and shape-polymorphic in ``L`` — a
    symbolic-length pad-and-concatenate does not lower to a static width."""
    dtype = z_col.dtype
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))  # (2ⁿᶜ,)
    # col_eq spans 2ⁿᶜ >= len(all_claims) in every layout (a column cap raises
    # n_c to log2_ceil(cap) alongside the pad), so the slice is always in-bounds;
    # the eq tail past the real columns multiplies zero-padded claims.
    return jnp.sum(col_eq[: all_claims.shape[0]] * all_claims)


def outer_sumcheck(
    dense: Array,
    indicator: Array,
    claim: Array,
    transcript: Transcript,
) -> tuple[Array, Array, Array, Transcript]:
    """Outer Hadamard sumcheck ``Σ_i D(i)·J̃(i) = claim``, LSB-first.

    Returns ``(round_polys (n,3), z_final (n,), dense_eval, transcript)`` where
    ``n = log2(len(dense))``. Folds even/odd pairs (``[0::2]``/``[1::2]``) one
    variable per round, observing each coefficient-form degree-2 round poly
    ``[s(0), claim-2·s(0)-s(∞), s(∞)]`` and sampling the next challenge; the
    point is the challenge list reversed (SP1's insert-at-front). ``dense_eval``
    is ``D(z_final)`` — the indicator factor is reproved by the inner sumcheck,
    not folded into the eval. Mirrors ``inner_sumcheck``'s LSB-first idiom over a
    flat Hadamard product (no branching program)."""
    state_a = dense
    state_b = indicator
    n_rounds = (state_a.shape[0] - 1).bit_length()
    ef = claim.dtype
    ef_limbs = efinfo(ef).degree
    two = jnp.array(2, ef)

    cur = claim
    polys: list[Array] = []
    challenges: list[Array] = []
    for _ in range(n_rounds):
        p0a, p1a = state_a[0::2], state_a[1::2]
        p0b, p1b = state_b[0::2], state_b[1::2]
        s0 = jnp.sum(p0a * p0b)
        s_inf = jnp.sum((p1a - p0a) * (p1b - p0b))
        coef = jnp.stack([s0, cur - two * s0 - s_inf, s_inf])

        # One extension challenge per variable; fused absorb+squeeze, so byte
        # for byte the same as observe + sample_challenge.
        transcript, raw = transcript.observe_and_sample(coef, ef_limbs)
        alpha = reinterpret_challenge(raw, ef)
        state_a = p0a + alpha * (p1a - p0a)
        state_b = p0b + alpha * (p1b - p0b)
        cur = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)

    dense_eval = state_a[0]
    z_final = jnp.stack(challenges)[::-1]
    return jnp.stack(polys), z_final, dense_eval, transcript


def _bp_all(
    buf: Array,
    z_row: Array,
    z_trace: Array,
    t_matrix: Array,
    num_bits: Any,
) -> Array:
    """Vectorize ``bp_eval_core`` over the column axis of ``buf`` ``(L,
    2·num_bits)``.

    ``num_bits`` (= n_d) is passed as a VALUE, not a static arg, so both the
    column count ``L`` and the prefix-bit width ``num_bits`` can be symbolic export
    dims: the half-split ``buf[:, :num_bits]`` / ``buf[:, num_bits:]`` slices at a
    symbolic midpoint and ``bp_eval_core``'s layer ``fori_loop`` (deriving its trip
    count from the ``num_bits``-wide prefix operands) takes a symbolic count."""
    return jax.vmap(
        lambda left, right: bp_eval_core(z_row, z_trace, left, right, t_matrix)
    )(buf[:, :num_bits], buf[:, num_bits:])


def inner_sumcheck_core(
    merged: Array,
    weights: Array,
    z_row: Array,
    z_trace: Array,
    transcript: Transcript,
    *,
    dtype: Any,
    num_bits: Any,
) -> tuple[Array, Array, Array, Transcript]:
    """Branching-program sumcheck over a prebuilt (merged, weights).

    Polymorphic in the column count L = merged.shape[0]: per-column work is a
    vmap + jnp.sum over the real columns (no padding), so L can be a symbolic
    export dim. The 2*num_bits round loop is unrolled (num_bits concrete) — one
    fused zorch.duplex_fs kernel per round. weights is the column-eq table
    col_eq[:L]; the caller keeps z_col at its true length (n_c, unpadded) so those
    weights are exact even when L is a symbolic dim."""
    n_vars = 2 * num_bits
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    one = jnp.ones((), dtype)
    two = jnp.array(2, dtype)
    ef_limbs = efinfo(dtype).degree

    def bp_all(buf: Array) -> Array:
        return _bp_all(buf, z_row, z_trace, t_matrix, num_bits)

    # claimed_sum = J̃(z_row, z_col, z_trace) = Σ_c eq(z_col,c)·bp_c — a jnp.sum,
    # not eval_jagged_mle's ~1700-deep trace-time unroll (which compiles abysmally).
    claimed_sum = jnp.sum(weights * bp_all(merged))

    # SP1's prove_jagged_evaluation absorbs the claimed J̃ value before the
    # rounds; its verifier re-absorbs it the same way (fractalyze/sp1-zorch#90).
    transcript = transcript.observe(claimed_sum)

    # Eliminate LSB-first (column n_vars-1 down to 0), unrolled so each round's
    # Fiat-Shamir absorb+squeeze lowers to its own fused zorch.duplex_fs kernel.
    # bits_i reads merged since the round's column is untouched until its own step
    # (merged == buf there).
    buf, claim, weights_c = merged, claimed_sum, weights
    polys: list[Array] = []
    challenges: list[Array] = []
    for round_idx in range(n_vars - 1, -1, -1):
        bits_i = merged[:, round_idx]
        eq0 = one - bits_i
        bp0 = bp_all(buf.at[:, round_idx].set(0))
        bp1 = bp_all(buf.at[:, round_idx].set(1))
        p0 = jnp.sum(weights_c * eq0 * bp0)
        p_inf = jnp.sum(weights_c * (bits_i - eq0) * (bp1 - bp0))
        coef = jnp.stack([p0, claim - two * p0 - p_inf, p_inf])

        # One extension challenge per variable; fused absorb+squeeze, so byte
        # for byte the same as observe + sample_challenge.
        transcript, raw = transcript.observe_and_sample(coef, ef_limbs)
        alpha = reinterpret_challenge(raw, dtype)
        buf = buf.at[:, round_idx].set(alpha)
        weights_c = weights_c * (alpha * bits_i + (one - alpha) * eq0)
        claim = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)
    return jnp.stack(polys), jnp.stack(challenges[::-1]), claimed_sum, transcript


def eval_round_core(
    offsets: Array,
    merged: Array,
    weights: Array,
    all_claims: Array,
    dense: Array,
    z_row: Array,
    z_col: Array,
    transcript: Transcript,
    *,
    dtype: Any,
) -> tuple[JaggedEvalMsg, Transcript]:
    """The whole eval-proof sumcheck over prebuilt column arrays, shape-polymorphic
    in the column count.

    All four column-indexed inputs share the column dim — ``offsets`` is
    ``(L+1, n_d)``, ``merged`` ``(L, 2·n_d)``, ``weights`` and ``all_claims``
    ``(L,)``. Every column-dependent step (the outer indicator's searchsorted
    gather, the outer ``Σ D·J̃`` Hadamard sumcheck, the inner branching-program
    sumcheck) runs over the REAL column count, so one ``jax.export`` binary serves
    every column count at real-size cost — no padding. The host builds ``offsets``
    / ``merged`` / ``weights`` from ``col_heights``; taking them as arrays here is
    what lets the column dim be symbolic. ``n_d = merged.shape[1] // 2``."""
    num_bits = merged.shape[1] // 2

    claim = outer_sumcheck_claim(all_claims, z_col)
    indicator = partial_eval_core(offsets, z_row, z_col, dense.shape[0])
    outer_polys, z_final, dense_eval, transcript = outer_sumcheck(
        dense, indicator, claim, transcript
    )
    inner_polys, inner_point, inner_claimed_sum, transcript = inner_sumcheck_core(
        merged,
        weights,
        z_row,
        z_final,
        transcript,
        dtype=dtype,
        num_bits=num_bits,
    )
    msg = JaggedEvalMsg(
        outer_sumcheck_claim=claim,
        outer_sumcheck_polys=outer_polys,
        outer_sumcheck_point=z_final,
        dense_eval=dense_eval,
        inner_sumcheck_polys=inner_polys,
        inner_point=inner_point,
        inner_claimed_sum=inner_claimed_sum,
    )
    return msg, transcript


def _eval_inputs(
    col_heights: Sequence[int], z_col: Array, dtype: Any, *, cap_l: int | None = None
) -> tuple[Array, Array, Array]:
    """Host-build the column arrays ``eval_round_core`` consumes: the canonical-limb
    offset tensor ``(L+1, n_d)``, the merged prefix-bit buffer ``(L, 2·n_d)``, and
    the column-eq weights ``col_eq[:L]``. ``n_d`` (= log-area tier) is the only
    static dim; the shape-polymorphic cores derive the rest from these shapes.

    With cap_l set every L-indexed output is laid at the fixed width cap_l: pad
    columns are empty ranges (0-height, so t_c = t_{c+1} → BP indicator 0) and the
    weights zero-extend, both fold-neutral, so the outputs stay byte-identical to
    the exact layout but ride one static shape."""
    heights = list(col_heights)
    true_l = len(heights)
    if cap_l is not None:
        if cap_l < true_l:
            raise ValueError(f"cap_l {cap_l} < column count {true_l}")
        heights = heights + [0] * (cap_l - true_l)  # empty-range pad (BP -> 0)
    l_max = len(heights)
    _, n_d = build_jagged_layout(heights, l_max, dtype)
    offsets = _offset_bit_tensor(heights, l_max, n_d, dtype)
    merged = merged_prefix_bits(heights, n_d, dtype=dtype)
    weights = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))[:true_l]
    if cap_l is not None:
        weights = jnp.concatenate([weights, jnp.zeros((cap_l - true_l,), dtype)])
    return offsets, merged, weights


class JaggedEvalRound(Round):
    """The jagged PCS evaluation-proof sumcheck as a composable IOP ``Round``.

    ``__call__`` maps ``(JaggedEvalInputs, transcript) -> (inputs, transcript,
    JaggedEvalMsg)`` so it sequences in ``ProveChain``. Runs the full sumcheck
    half: the outer Hadamard sumcheck ``Σ D·J̃`` over the committed dense buffer
    (round polys + ``dense_eval``), whose folded point ``z_final`` then feeds the
    inner branching-program sumcheck reproving ``J̃(z_row, z_col, z_final)``. See
    the module docstring for why both are bespoke loops, not ``SumcheckRound``s.

    Host-prepares the column arrays from ``col_heights`` then defers to
    ``eval_round_core`` (shape-polymorphic in the column count)."""

    def __init__(self, *, dtype: Any) -> None:
        self._dtype = dtype

    def __call__(
        self, carry: JaggedEvalInputs, transcript: Transcript
    ) -> tuple[JaggedEvalInputs, Transcript, JaggedEvalMsg]:
        z_col = carry.z_col
        all_claims = carry.all_claims
        if carry.cap_l is not None:
            # n_c = ⌈log2 L⌉ tracks the true column count, so cap it alongside L.
            # PREPEND the extra challenge coords as zeros: they are the high-order
            # column bits, and every real column index c < 2ⁿᶜ has those bits 0,
            # so `eq(0, 0) = 1` leaves col_eq[c] unchanged -- only the masked pad
            # columns see the new coords. Byte-identical, one static z_col shape.
            cap_nc = log2_ceil_usize(carry.cap_l)
            npad = cap_nc - z_col.shape[0]
            if npad:
                z_col = jnp.concatenate([jnp.zeros((npad,), self._dtype), z_col])
            if all_claims.shape[0] < carry.cap_l:
                pad = carry.cap_l - all_claims.shape[0]
                all_claims = jnp.concatenate(
                    [all_claims, jnp.zeros((pad,), self._dtype)]
                )
        offsets, merged, weights = _eval_inputs(
            carry.col_heights, z_col, self._dtype, cap_l=carry.cap_l
        )
        msg, transcript = eval_round_core(
            offsets,
            merged,
            weights,
            all_claims,
            carry.dense,
            carry.z_row,
            z_col,
            transcript,
            dtype=self._dtype,
        )
        return carry, transcript, msg


__all__ = [
    "JaggedEvalInputs",
    "JaggedEvalMsg",
    "JaggedEvalRound",
    "assemble_columns",
    "merged_prefix_bits",
    "outer_sumcheck_claim",
    "outer_sumcheck",
    "inner_sumcheck_core",
    "eval_round_core",
    "sample_z_col",
]
