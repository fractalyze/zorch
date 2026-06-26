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
    partial_eval,
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


@dataclass(frozen=True)
class JaggedEvalMsg:
    """Proof message: the outer Hadamard sumcheck (initial column claim, its
    coefficient-form round polys, the folded point ``z_final``, and
    ``dense_eval = D(z_final)``) and the inner branching-program sumcheck
    transcript (coefficient-form round polys, the folded point, the reproved
    claim)."""

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
    """``Σ_c eq(z_col, c)·claim[c]`` over the padded 2^⌈log L⌉ column hypercube."""
    dtype = z_col.dtype
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))  # (2^n_c,)
    width = col_eq.shape[0]
    padded = jnp.concatenate(
        [all_claims, jnp.zeros((width - all_claims.shape[0],), dtype=dtype)], axis=0
    )
    return jnp.sum(col_eq * padded)


def build_outer_indicator(
    col_heights: Sequence[int],
    z_row: Array,
    z_col: Array,
    target_size: int,
    *,
    dtype: Any,
) -> Array:
    """Materialize ``J̃(z_row, z_col, ·)`` over the dense area ``[0, target_size)``.

    ``partial_eval`` allocates ``2^n_d`` (``n_d`` = prefix-bit width, which can
    exceed the dense area's address width when the total area is itself a power
    of two), reading its scatter offsets from the canonical-limb tensor; every
    nonzero entry lands below ``total_area <= target_size``, so the tail is zero
    and slicing to ``target_size`` recovers the indicator over the dense area."""
    heights = list(col_heights)
    l_max = len(heights)
    _, cfg = build_jagged_layout(heights, l_max, z_row.shape[0], dtype)
    offsets = _offset_bit_tensor(heights, l_max, cfg)
    return partial_eval(offsets, z_row, z_col, cfg=cfg)[:target_size]


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


def inner_sumcheck(
    col_heights: Sequence[int],
    z_row: Array,
    z_col: Array,
    z_trace: Array,
    transcript: Transcript,
    *,
    dtype: Any,
) -> tuple[Array, Array, Array, Transcript]:
    """Reprove ``J̃(z_row, z_col, z_trace)`` via the branching-program sumcheck.

    Returns ``(round_polys (n_vars,3), inner_point (n_vars,), claimed_sum,
    transcript)``. ``n_vars = 2·n_d`` over the merged prefix-bit buffer,
    eliminated LSB-first (buffer column ``n_vars-1`` down to ``0``); each round's
    coefficient-form poly is observed and the next challenge sampled."""
    heights = list(col_heights)
    l_max = len(heights)
    _, cfg = build_jagged_layout(heights, l_max, z_row.shape[0], dtype)

    # num_bits = cfg.n_d holds prefix sums up to the total area; the merged
    # buffer carries bits(t_c) ‖ bits(t_{c+1}) -> n_vars = 2·n_d.
    num_bits = cfg.n_d
    n_vars = 2 * num_bits
    # The BP indexes over n_d prefix bits, not z_trace's length — the layer loop
    # must cover all prefix bits or it drops the MSB (matches eval_jagged_mle's
    # num_vars = max(n_r, n_d)).
    bp_num_vars = max(cfg.n_r, cfg.n_d)
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)

    merged = merged_prefix_bits(heights, num_bits, dtype=dtype)

    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))
    weights = col_eq[:l_max]
    one = jnp.ones((), dtype)
    two = jnp.array(2, dtype)
    ef_limbs = efinfo(dtype).degree

    @jax.jit
    def bp_all(buf: Array) -> Array:
        return jax.vmap(
            lambda left, right: bp_eval_core(
                z_row, z_trace, left, right, t_matrix, bp_num_vars
            )
        )(buf[:, :num_bits], buf[:, num_bits:])

    # claimed_sum = J̃(z_row, z_col, z_trace) = Σ_c eq(z_col,c)·bp_c. Computed via
    # jnp.sum (CPU EF reduce works) rather than eval_jagged_mle's trace-time
    # 1726-deep unroll, which compiles abysmally.
    claimed_sum = jnp.sum(weights * bp_all(merged))

    # SP1's prove_jagged_evaluation absorbs the claimed J̃ value before the
    # rounds; its verifier re-absorbs it the same way (fractalyze/sp1-zorch#90).
    transcript = transcript.observe(claimed_sum)

    buf = merged
    claim = claimed_sum
    polys: list[Array] = []
    challenges: list[Array] = []
    for round_idx in range(n_vars - 1, -1, -1):
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
        polys.append(coef)
        challenges.append(alpha)

    return jnp.stack(polys), jnp.stack(challenges)[::-1], claimed_sum, transcript


class JaggedEvalRound(Round):
    """The jagged PCS evaluation-proof sumcheck as a composable IOP ``Round``.

    ``__call__`` maps ``(JaggedEvalInputs, transcript) -> (inputs, transcript,
    JaggedEvalMsg)`` so it sequences in ``ProveChain``. Runs the full sumcheck
    half: the outer Hadamard sumcheck ``Σ D·J̃`` over the committed dense buffer
    (round polys + ``dense_eval``), whose folded point ``z_final`` then feeds the
    inner branching-program sumcheck reproving ``J̃(z_row, z_col, z_final)``. See
    the module docstring for why both are bespoke loops, not ``SumcheckRound``s."""

    def __init__(self, *, dtype: Any) -> None:
        self._dtype = dtype

    def __call__(
        self, carry: JaggedEvalInputs, transcript: Transcript
    ) -> tuple[JaggedEvalInputs, Transcript, JaggedEvalMsg]:
        claim = outer_sumcheck_claim(carry.all_claims, carry.z_col)
        indicator = build_outer_indicator(
            carry.col_heights,
            carry.z_row,
            carry.z_col,
            carry.dense.shape[0],
            dtype=self._dtype,
        )
        outer_polys, z_final, dense_eval, transcript = outer_sumcheck(
            carry.dense, indicator, claim, transcript
        )
        inner_polys, inner_point, inner_claimed_sum, transcript = inner_sumcheck(
            carry.col_heights,
            carry.z_row,
            carry.z_col,
            z_final,
            transcript,
            dtype=self._dtype,
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
        return carry, transcript, msg


__all__ = [
    "JaggedEvalInputs",
    "JaggedEvalMsg",
    "JaggedEvalRound",
    "assemble_columns",
    "merged_prefix_bits",
    "outer_sumcheck_claim",
    "build_outer_indicator",
    "outer_sumcheck",
    "inner_sumcheck",
    "sample_z_col",
]
