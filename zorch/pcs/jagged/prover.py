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
``zerocheck/jagged.py``. (Consequently it does not emit the ``zorch.sumcheck``
composite — the dense-only SVO / register-resident codegen does not apply; the
jagged equivalent is separate GPU-codegen work.)

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

from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    _offset_bit_tensor,
    bp_eval_core,
    build_jagged_layout,
    build_prefix_sums,
    msb_first_bits,
    partial_eval_core,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round
from zorch.transcript import Transcript, sample_challenge
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
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))  # (2^n_c,)
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

        # SP1 binds each variable with one extension element (its
        # ``sample_ext_element``) — the shared ``sample_challenge`` rule.
        transcript = transcript.observe(coef)
        transcript, alpha = sample_challenge(transcript, ef, ef_limbs)
        state_a = p0a + alpha * (p1a - p0a)
        state_b = p0b + alpha * (p1b - p0b)
        cur = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)

    dense_eval = state_a[0]
    z_final = jnp.stack(challenges)[::-1]
    return jnp.stack(polys), z_final, dense_eval, transcript


def outer_sumcheck_scan(
    dense: Array,
    indicator: Array,
    claim: Array,
    transcript: Transcript,
    n_rounds: Any,
) -> tuple[Array, Array, Array, Transcript]:
    """Fixed-width-mask form of ``outer_sumcheck`` so the round count
    ``n = log2(len(dense))`` can be a symbolic ``jax.export`` dim.

    A ``lax.scan`` can't carry the halving state's shrinking shape, so the buffers
    stay full width with the live data front-packed (``zeros(M).at[:half].set``)
    and the dead tail masked out of each round-poly sum (the same device as
    ``zorch.sumcheck.prover.prove``, here LSB-first to match the jagged schedule).
    Byte-identical to ``outer_sumcheck``.

    TRADEOFF: full-width work every round (~n× the halving's 2·M total). That is
    the necessary cost of one symbolic binary over a halving fold; the live
    ``JaggedEvalRound`` keeps the real-halving ``outer_sumcheck``. ``n_rounds`` is
    passed in because ``log2`` is not polynomial in the dense length, so under
    export it is its own symbolic dim (paired with the length at the call)."""
    ef = claim.dtype
    ef_limbs = efinfo(ef).degree
    two = jnp.array(2, ef)
    full = dense.shape[0]
    half = full // 2
    # Lift dense (base field) to EF up front: the halving fold promotes the state
    # BF->EF after round 0, but a scan carry must keep one dtype. The lift is the
    # constant-term embedding, so the round-0 BF*EF product is byte-identical.
    dense_ef = dense * jnp.ones((), ef)

    def _round(carry: Any, _: Array) -> tuple[Any, tuple[Array, Array]]:
        a, b, cur, transcript, valid_pairs = carry
        # Pair via reshape (full -> (half, 2)), not a[0::2]/[1::2]: a strided slice
        # of a symbolic length lowers to (full+1)//2, which the export solver can't
        # prove equals half. Row j = [arr[2j], arr[2j+1]] == the even/odd split.
        pa, pb = a.reshape(half, 2), b.reshape(half, 2)
        even_a, odd_a = pa[:, 0], pa[:, 1]
        even_b, odd_b = pb[:, 0], pb[:, 1]
        valid = (jnp.arange(half) < valid_pairs).astype(ef)
        s0 = jnp.sum(even_a * even_b * valid)
        s_inf = jnp.sum((odd_a - even_a) * (odd_b - even_b) * valid)
        coef = jnp.stack([s0, cur - two * s0 - s_inf, s_inf])
        transcript = transcript.observe(coef)
        transcript, alpha = sample_challenge(transcript, ef, ef_limbs)
        fold_a = even_a + alpha * (odd_a - even_a)
        fold_b = even_b + alpha * (odd_b - even_b)
        a = jnp.zeros((full,), ef).at[:half].set(fold_a)
        b = jnp.zeros((full,), ef).at[:half].set(fold_b)
        cur = eval_coeffs(coef, alpha)
        return (a, b, cur, transcript, valid_pairs // 2), (coef, alpha)

    vp0 = jnp.asarray(half, jnp.int32)
    (a_f, _, _, transcript, _), (polys, challenges) = jax.lax.scan(
        _round,
        (dense_ef, indicator, claim, transcript, vp0),
        jnp.arange(n_rounds, dtype=jnp.int32),
    )
    return polys, challenges[::-1], a_f[0], transcript


def _bp_all(
    buf: Array,
    z_row: Array,
    z_trace: Array,
    t_matrix: Array,
    bp_num_vars: Any,
    num_bits: Any,
) -> Array:
    """Vectorize ``bp_eval_core`` over the column axis of ``buf`` ``(L,
    2·num_bits)``.

    ``num_bits`` (= n_d) and ``bp_num_vars`` (= max(n_r, n_d), the BP layer count)
    are passed as VALUES, not static args, so both the column count ``L`` and the
    prefix-bit width ``num_bits`` can be symbolic export dims: the half-split
    ``buf[:, :num_bits]`` / ``buf[:, num_bits:]`` slices at a symbolic midpoint and
    ``bp_eval_core``'s layer ``fori_loop`` takes a symbolic trip count."""
    return jax.vmap(
        lambda left, right: bp_eval_core(
            z_row, z_trace, left, right, t_matrix, bp_num_vars
        )
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
    """The branching-program sumcheck over a prebuilt ``(merged, weights)``,
    shape-polymorphic in BOTH the column count ``merged.shape[0]`` and the
    prefix-bit width ``num_bits`` (= n_d).

    Column axis: per-column work is a ``vmap`` + ``jnp.sum`` over ``merged``'s real
    columns (no padding), mirroring ``stacked_basefold_open``'s symbolic ``K``.
    ``n_d`` axis: the round loop is a ``lax.scan`` over ``n_vars = 2·num_bits``
    rounds (fixed-shape carry — the buffer is bound in place), so the round count
    can be a symbolic export dim; ``bp_num_vars = max(n_r, n_d)`` is the BP layer
    count, derived as a value so it stays symbolic-safe. ``weights`` is the
    column-eq table ``col_eq[:L]`` the caller derives from ``z_col`` (``z_col``
    stays at the real ``n_c``, so SP1 byte-match is preserved)."""
    n_vars = 2 * num_bits
    bp_num_vars = jnp.maximum(z_row.shape[0], num_bits)
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    one = jnp.ones((), dtype)
    two = jnp.array(2, dtype)
    ef_limbs = efinfo(dtype).degree

    def bp_all(buf: Array) -> Array:
        return _bp_all(buf, z_row, z_trace, t_matrix, bp_num_vars, num_bits)

    # claimed_sum = J̃(z_row, z_col, z_trace) = Σ_c eq(z_col,c)·bp_c. Computed via
    # jnp.sum (CPU EF reduce works) rather than eval_jagged_mle's trace-time
    # 1726-deep unroll, which compiles abysmally.
    claimed_sum = jnp.sum(weights * bp_all(merged))

    # SP1's prove_jagged_evaluation absorbs the claimed J̃ value before the
    # rounds; its verifier re-absorbs it the same way (fractalyze/sp1-zorch#90).
    transcript = transcript.observe(claimed_sum)

    # Eliminate LSB-first, buffer column n_vars-1 down to 0, as a lax.scan so the
    # round count (= n_vars = 2·n_d) can be a symbolic export dim. The carry keeps
    # a FIXED shape (the buffer is bound in place, never shrunk), which scan
    # requires; bits_i reads `merged` (the round's column is untouched until its
    # own step, so merged == buf there). Stacked outputs land in scan order
    # (n_vars-1 .. 0) — same as the unrolled append order.
    def _round(carry: Any, round_idx: Array) -> tuple[Any, tuple[Array, Array]]:
        buf, claim, weights, transcript = carry
        bits_i = merged[:, round_idx]
        eq0 = one - bits_i
        bp0 = bp_all(buf.at[:, round_idx].set(0))
        bp1 = bp_all(buf.at[:, round_idx].set(1))
        p0 = jnp.sum(weights * eq0 * bp0)
        p_inf = jnp.sum(weights * (bits_i - eq0) * (bp1 - bp0))
        coef = jnp.stack([p0, claim - two * p0 - p_inf, p_inf])

        # One extension element per variable, as in ``outer_sumcheck``.
        transcript = transcript.observe(coef)
        transcript, alpha = sample_challenge(transcript, dtype, ef_limbs)
        buf = buf.at[:, round_idx].set(alpha)
        weights = weights * (alpha * bits_i + (one - alpha) * eq0)
        claim = eval_coeffs(coef, alpha)
        return (buf, claim, weights, transcript), (coef, alpha)

    round_ids = jnp.arange(n_vars, dtype=jnp.int32)[::-1]
    (_, _, _, transcript), (polys, challenges) = jax.lax.scan(
        _round, (merged, claimed_sum, weights, transcript), round_ids
    )
    return polys, challenges[::-1], claimed_sum, transcript


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
    col_heights: Sequence[int], z_col: Array, dtype: Any
) -> tuple[Array, Array, Array]:
    """Host-build the column arrays ``eval_round_core`` consumes: the canonical-limb
    offset tensor ``(L+1, n_d)``, the merged prefix-bit buffer ``(L, 2·n_d)``, and
    the column-eq weights ``col_eq[:L]``. ``n_d`` (= log-area tier) is the only
    static dim; the shape-polymorphic cores derive the rest from these shapes."""
    heights = list(col_heights)
    l_max = len(heights)
    _, n_d = build_jagged_layout(heights, l_max, dtype)
    offsets = _offset_bit_tensor(heights, l_max, n_d, dtype)
    merged = merged_prefix_bits(heights, n_d, dtype=dtype)
    weights = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))[:l_max]
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
        offsets, merged, weights = _eval_inputs(
            carry.col_heights, carry.z_col, self._dtype
        )
        msg, transcript = eval_round_core(
            offsets,
            merged,
            weights,
            carry.all_claims,
            carry.dense,
            carry.z_row,
            carry.z_col,
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
    "outer_sumcheck_scan",
    "inner_sumcheck_core",
    "eval_round_core",
    "sample_z_col",
]
