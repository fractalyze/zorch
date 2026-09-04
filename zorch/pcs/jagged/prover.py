# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule jagged evaluation-proof sumcheck as one recurrence step.

``prove_jagged_eval`` is the sumcheck half of SP1's jagged evaluation phase;
a PCS stage drives it alongside its verifier dual and the remaining opening
work. It reproves SP1's jagged PCS
opening sumchecks byte-identically: the OUTER Hadamard sumcheck
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

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import efinfo

from zorch.challenge import ChallengePolicy
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
from zorch.transcript import Transcript, sample_challenge
from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class JaggedEvalInputs:
    """Input to ``prove_jagged_eval``: the committed columns' jagged layout plus
    the points the upstream rounds fixed.

    ``col_heights`` is the per-unit-column height list and ``all_claims`` the
    matching ``(L,)`` per-column GKR openings (see ``assemble_columns``).
    ``dense`` is the combined committed dense buffer ``D`` (both rounds' raw
    packed columns concatenated, padded to ``2^n``) over which the outer
    Hadamard sumcheck runs; the outer point ``z_final`` it produces feeds the
    inner sumcheck, so it is not carried in."""

    col_heights: tuple[int, ...]
    all_claims: Array
    z_row: Array
    z_col: Array
    dense: Array


@partial(
    frx.tree_util.register_dataclass,
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

    A registered pytree so it crosses the ``eval_round_core`` ``@frx.jit`` /
    ``frx.export`` boundary (mirrors ``open.py``'s ``StackedOpenProof``)."""

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
    col_heights = assemble_col_heights(row_counts_rounds, column_counts_rounds)
    claim_blocks: list[Array] = []
    for ccs, claims_r in zip(column_counts_rounds, column_claims_rounds, strict=True):
        n_pad = int(ccs[-2]) + int(ccs[-1])
        claim_blocks.append(fnp.asarray(claims_r, dtype=dtype))
        if n_pad:
            claim_blocks.append(fnp.zeros((n_pad,), dtype=dtype))
    return col_heights, fnp.concatenate(claim_blocks, axis=0)


def assemble_col_heights(
    row_counts_rounds: Sequence[Sequence[int]],
    column_counts_rounds: Sequence[Sequence[int]],
) -> list[int]:
    """The per-unit-column height list alone — host ints, no claim arrays, so a
    consumer can derive the layout eagerly and defer the claim assembly to a
    jitted body (``assemble_columns`` delegates here)."""
    col_heights: list[int] = []
    for rcs, ccs in zip(row_counts_rounds, column_counts_rounds, strict=True):
        if len(ccs) < 2:
            raise ValueError(
                f"each round needs the trailing (stacking-dummy, leftover) "
                f"column-count pair; got {len(ccs)} counts"
            )
        for rc, cc in zip(rcs, ccs, strict=True):
            col_heights.extend([int(rc)] * int(cc))
    return col_heights


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
    z_col = fnp.stack(parts) if parts else fnp.zeros((0,), dtype)
    return transcript, z_col


def merged_prefix_bits(
    col_heights: Sequence[int], num_bits: int, *, dtype: Any
) -> Array:
    """The ``(L, 2·num_bits)`` merged prefix-bit buffer ``bits(t_c) ‖
    bits(t_{c+1})`` — the branching-program input both the inner sumcheck and
    its verifier leaf check read."""
    prefix_int = build_prefix_sums(list(col_heights))
    bits = msb_first_bits(prefix_int, num_bits)
    return fnp.asarray(np.concatenate([bits[:-1], bits[1:]], axis=1), dtype=dtype)


def outer_sumcheck_claim(all_claims: Array, z_col: Array) -> Array:
    """``Σ_c eq(z_col, c)·claim[c]`` over the real columns of the 2^⌈log L⌉ hypercube.

    The eq tail past ``L`` would multiply zero-padded claims, so summing only the
    real columns (``col_eq[:L]``) is identical and shape-polymorphic in ``L`` — a
    symbolic-length pad-and-concatenate does not lower to a static width."""
    dtype = z_col.dtype
    col_eq = expand_eq_to_hypercube(z_col, fnp.ones((), dtype))  # (2ⁿᶜ,)
    return fnp.sum(col_eq[: all_claims.shape[0]] * all_claims)


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
    ef_challenges = ChallengePolicy(ef)
    two = fnp.array(2, ef)

    cur = claim
    polys: list[Array] = []
    challenges: list[Array] = []
    for _ in range(n_rounds):
        p0a, p1a = state_a[0::2], state_a[1::2]
        p0b, p1b = state_b[0::2], state_b[1::2]
        s0 = fnp.sum(p0a * p0b)
        s_inf = fnp.sum((p1a - p0a) * (p1b - p0b))
        coef = fnp.stack([s0, cur - two * s0 - s_inf, s_inf])

        # One extension challenge per variable; fused absorb+squeeze, so byte
        # for byte the same as observe + sample_challenge.
        transcript, alpha = ef_challenges.observe_and_sample(transcript, coef)
        state_a = p0a + alpha * (p1a - p0a)
        state_b = p0b + alpha * (p1b - p0b)
        cur = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)

    dense_eval = state_a[0]
    z_final = fnp.stack(challenges)[::-1]
    return fnp.stack(polys), z_final, dense_eval, transcript


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
    return frx.vmap(
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
    vmap + fnp.sum over the real columns (no padding), so L can be a symbolic
    export dim. The 2*num_bits round loop is unrolled (num_bits concrete) — one
    fused zorch.duplex_fs kernel per round. weights is the column-eq table
    col_eq[:L]; the caller keeps z_col at its true length (n_c, unpadded) so those
    weights are exact even when L is a symbolic dim."""
    n_vars = 2 * num_bits
    t_matrix = fnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    one = fnp.ones((), dtype)
    two = fnp.array(2, dtype)
    ef_challenges = ChallengePolicy(dtype)

    def bp_all(buf: Array) -> Array:
        return _bp_all(buf, z_row, z_trace, t_matrix, num_bits)

    # claimed_sum = J̃(z_row, z_col, z_trace) = Σ_c eq(z_col,c)·bp_c — a fnp.sum,
    # not eval_jagged_mle's ~1700-deep trace-time unroll (which compiles abysmally).
    claimed_sum = fnp.sum(weights * bp_all(merged))

    # SP1's prove_jagged_evaluation absorbs the claimed J̃ value before the
    # rounds; its verifier re-absorbs it the same way.
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
        p0 = fnp.sum(weights_c * eq0 * bp0)
        p_inf = fnp.sum(weights_c * (bits_i - eq0) * (bp1 - bp0))
        coef = fnp.stack([p0, claim - two * p0 - p_inf, p_inf])

        # One extension challenge per variable; fused absorb+squeeze, so byte
        # for byte the same as observe + sample_challenge.
        transcript, alpha = ef_challenges.observe_and_sample(transcript, coef)
        buf = buf.at[:, round_idx].set(alpha)
        weights_c = weights_c * (alpha * bits_i + (one - alpha) * eq0)
        claim = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)
    return fnp.stack(polys), fnp.stack(challenges[::-1]), claimed_sum, transcript


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
    sumcheck) runs over the REAL column count, so one ``frx.export`` binary serves
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


def eval_column_arrays(
    col_heights: Sequence[int], *, dtype: Any
) -> tuple[Array, Array]:
    """Host-build the two height-dependent column arrays ``eval_round_core``
    consumes: the offset tensor ``(L+1, n_d)`` and the merged prefix-bit
    buffer ``(L, 2·n_d)``. Heights live in the array VALUES, so a jitted
    consumer taking these as traced arguments keys its compile on the
    ``(L, n_d)`` class alone."""
    heights = list(col_heights)
    l_max = len(heights)
    _, n_d = build_jagged_layout(heights, l_max, dtype)
    offsets = _offset_bit_tensor(heights, l_max, n_d, dtype)
    merged = merged_prefix_bits(heights, n_d, dtype=dtype)
    return offsets, merged


def _eval_inputs(
    col_heights: Sequence[int], z_col: Array, dtype: Any
) -> tuple[Array, Array, Array]:
    """The column arrays plus the column-eq weights ``col_eq[:L]`` — the eager
    eager form, where ``z_col`` is already concrete."""
    l_max = len(col_heights)
    # col_eq = expand_eq(z_col) has 2^len(z_col) entries; too few z_col vars would
    # silently truncate weights[:l_max] below l_max and mismatch merged downstream.
    if z_col.shape[0] < log2_ceil_usize(l_max):
        raise ValueError(
            f"z_col has {z_col.shape[0]} variables, too few for {l_max} columns "
            f"(need ≥ {log2_ceil_usize(l_max)})"
        )
    offsets, merged = eval_column_arrays(col_heights, dtype=dtype)
    weights = expand_eq_to_hypercube(z_col, fnp.ones((), dtype))[:l_max]
    return offsets, merged, weights


def prove_jagged_eval(
    inputs: JaggedEvalInputs, transcript: Transcript, *, dtype: Any
) -> tuple[JaggedEvalMsg, Transcript]:
    """The jagged PCS evaluation sumchecks over a `JaggedEvalInputs`.

    Runs the full sumcheck half: the outer Hadamard sumcheck ``Sum D*J~`` over
    the committed dense buffer (round polys + ``dense_eval``), whose folded
    point ``z_final`` then feeds the inner branching-program sumcheck reproving
    ``J~(z_row, z_col, z_final)``. See the module docstring for why both are
    bespoke loops, not ``SumcheckRound``s.

    A function rather than a round: it reduces no carry — the layout it is
    handed is the layout it proves — so there is nothing for a recurrence to
    thread. Host-prepares the column arrays from ``col_heights`` then defers to
    ``eval_round_core`` (shape-polymorphic in the column count).
    """
    offsets, merged, weights = _eval_inputs(inputs.col_heights, inputs.z_col, dtype)
    return eval_round_core(
        offsets,
        merged,
        weights,
        inputs.all_claims,
        inputs.dense,
        inputs.z_row,
        inputs.z_col,
        transcript,
        dtype=dtype,
    )


__all__ = [
    "JaggedEvalInputs",
    "JaggedEvalMsg",
    "prove_jagged_eval",
    "assemble_col_heights",
    "assemble_columns",
    "eval_column_arrays",
    "merged_prefix_bits",
    "outer_sumcheck_claim",
    "outer_sumcheck",
    "inner_sumcheck_core",
    "eval_round_core",
    "sample_z_col",
]
