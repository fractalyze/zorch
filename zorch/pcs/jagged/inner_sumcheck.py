# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Inner jagged-assist sumcheck — open + verify J̃(z_row, z_col, z_final).

The outer Hadamard sumcheck reduces to a point z_final and needs the jagged MLE
value J̃(z_row, z_col, z_final) = Σ_c eq(z_col, c)·h(z_row, z_final; t_c, t_{c+1}),
an O(l_max)-term sum over all columns. This inner sumcheck reduces that L-sum to a
single branching-program leaf the verifier computes, by running a degree-2
sumcheck over the merged-prefix-bit hypercube x = (x_left ‖ x_right) (each half
num_bits = n_d bits, total 2·n_d rounds), summand eq_weighted(x)·h_bp(x) where

    eq_weighted(x) = Σ_c eq(z_col, c)·δ(x, t_c ‖ t_{c+1})

is a sum of L delta functions. Summing over a boolean free variable therefore
collapses to L terms per round, so the prover runs in O(n_vars·l_max·BP) rather
than O(2^n_vars). At the reduced point final_point = (z_left ‖ z_right) the
verifier checks

    (Σ_c eq(z_col, c)·eval_eq(t_c ‖ t_{c+1}, final_point))·h(z_row, z_final;
        z_left, z_right) == final_eval.

Round polynomials ride in monomial-coefficient form [c0, c1, c2]: c0 = s(0),
c2 = s(∞) (leading), c1 = claim − 2·c0 − c2 (from s(0)+s(1) = claim). EF sums use
a trace-time functools.reduce over the static l_max — jnp.sum over an extension
field aborts the ZKX backend (see docs/poly.md), so the column reduction is
unrolled.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import partial
from functools import reduce as _reduce

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array
from jax.tree_util import register_dataclass

from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    JaggedStaticConfig,
    bp_eval_core,
    eval_jagged_mle,
)
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.prove import RoundMsg, fold_rounds
from zorch.round import Round
from zorch.transcript import Transcript


def _ef_sum(terms: list[Array]) -> Array:
    """Σ over a static-length list of extension-field scalars at trace time.

    jnp.sum over an extension-field array aborts the ZKX backend, so the column
    reduction is a Python-level fold over the static l_max (same semantics)."""
    return _reduce(operator.add, terms)


@partial(
    register_dataclass,
    data_fields=[
        "weights",
        "buf",
        "merged_bits",
        "claim",
        "rounds_left",
        "z_row",
        "z_final",
        "t_matrix",
    ],
    meta_fields=["num_vars"],
)
@dataclass(frozen=True)
class _InnerState:
    """Carry for the jagged-assist round.

    weights: (l_max,) running delta weights w_c = eq(z_col, c)·∏_{j done} eq(α_j,
        m_c[j]) — the only thing that folds across rounds.
    buf: (l_max, 2·num_bits) merged-prefix bit tables, column `round_idx`
        overwritten by α each round so the next round's BP reads the bound value.
    merged_bits: (l_max, 2·num_bits) immutable original bits, source of bits_i.
    claim: running sumcheck claim s_prev(α_prev).
    rounds_left: int32 countdown; the buffer column eliminated this round is
        `rounds_left - 1` (LSB-first over the 2·num_bits-wide buffer).
    z_row, z_final, t_matrix, num_vars: fixed branching-program arguments.

    A registered pytree (num_vars static) so it threads through the per-round
    `@jit` — each round shares the carry's shapes, so the round body compiles once
    and `fold_rounds`'s nine remaining calls reuse it, instead of unrolling ten
    sponge permutes into one graph (a multi-minute XLA compile on the ZKX CPU
    backend)."""

    weights: Array
    buf: Array
    merged_bits: Array
    claim: Array
    rounds_left: Array
    z_row: Array
    z_final: Array
    t_matrix: Array
    num_vars: int


@jax.jit
def _jagged_assist_step(
    state: _InnerState, transcript: Transcript
) -> tuple[_InnerState, Transcript, RoundMsg]:
    """One jitted jagged-assist round body — see `JaggedAssistRound`. The carry
    shapes are round-invariant, so this compiles once and `fold_rounds` reuses it
    across all 2·n_d rounds (the sponge permute is not re-unrolled per round)."""
    dtype = state.z_row.dtype
    num_bits = state.merged_bits.shape[1] // 2
    n_cols = state.buf.shape[0]
    one = jnp.ones([], dtype=dtype)
    col_shape = (n_cols,)

    # The buffer column eliminated this round, LSB-first over the 2·num_bits
    # merged-bit buffer (the buffer stores bits MSB-first, so eliminate from the
    # high index down).
    round_idx = state.rounds_left - 1

    # Branching program at this column = 0 and = 1, over all L delta columns.
    buf_0 = state.buf.at[:, round_idx].set(jnp.zeros(col_shape, dtype=dtype))
    all_bp_0 = jax.vmap(
        lambda pl, pr: bp_eval_core(
            state.z_row, state.z_final, pl, pr, state.t_matrix, state.num_vars
        )
    )(buf_0[:, :num_bits], buf_0[:, num_bits:])
    buf_1 = state.buf.at[:, round_idx].set(jnp.ones(col_shape, dtype=dtype))
    all_bp_1 = jax.vmap(
        lambda pl, pr: bp_eval_core(
            state.z_row, state.z_final, pl, pr, state.t_matrix, state.num_vars
        )
    )(buf_1[:, :num_bits], buf_1[:, num_bits:])

    bits_i = state.merged_bits[:, round_idx]  # (l_max,) the column's true bit
    eq_0 = one - bits_i

    # p_0 = s(0) = Σ_c w_c·eq_0·bp_0;
    # p_inf = s(∞) = Σ_c w_c·(bits − eq_0)·(bp_1 − bp_0).
    # EF reduce via trace-time fold over the static l_max.
    p_0 = _ef_sum([state.weights[c] * eq_0[c] * all_bp_0[c] for c in range(n_cols)])
    p_inf = _ef_sum(
        [
            state.weights[c] * (bits_i[c] - eq_0[c]) * (all_bp_1[c] - all_bp_0[c])
            for c in range(n_cols)
        ]
    )
    coef_poly = jnp.stack([p_0, state.claim - 2 * p_0 - p_inf, p_inf])

    # Sample one base-field challenge and fold the EF state with it directly —
    # the standard zorch idiom, matching `SumcheckRound`'s observe-then-sample
    # order so prover and verifier transcripts stay aligned. The sponge squeezes
    # a Montgomery-form base scalar; cast it to the EF's canonical base field so
    # base·EF promotes (mixed Montgomery-ness has no promotion path), the same
    # base/EF fold `zorch.coding.fri` runs.
    transcript, r = transcript.observe_and_sample(coef_poly, 1)
    alpha = r[0].astype(zk_dtypes.efinfo(dtype).base_field_dtype)

    # Fold: bind this column to α and update the running weights/claim.
    new_buf = state.buf.at[:, round_idx].set(jnp.broadcast_to(alpha, col_shape))
    new_weights = state.weights * (alpha * bits_i + (one - alpha) * (one - bits_i))
    # claim' = s(α) from monomial coefs [c0, c1, c2] (Horner).
    new_claim = coef_poly[0] + alpha * (coef_poly[1] + alpha * coef_poly[2])

    new_state = _InnerState(
        weights=new_weights,
        buf=new_buf,
        merged_bits=state.merged_bits,
        claim=new_claim,
        rounds_left=round_idx,
        z_row=state.z_row,
        z_final=state.z_final,
        t_matrix=state.t_matrix,
        num_vars=state.num_vars,
    )
    return new_state, transcript, RoundMsg(coef_poly, alpha)


class JaggedAssistRound(Round):
    """One degree-2 jagged-assist sumcheck round (prover side).

    Maps (state, transcript) -> (state, transcript, RoundMsg). Per round:
    eliminate the next merged-bit variable by evaluating the branching program at
    the column's 0/1 settings over the L deltas, build the degree-2 round poly
    (monomial [p_0, claim - 2*p_0 - p_inf, p_inf]), observe + sample alpha, then
    fold — bind that column to alpha, update the running weights by eq(alpha,
    bit_c), and reduce the claim to s(alpha). The fold lands in the emitted state,
    so the round is self-contained and `fold_rounds` chains it straight through
    with no prior-challenge plumbing. Delegates to the jitted `_jagged_assist_step`
    so the round body compiles once and is reused across rounds."""

    def __call__(
        self, state: _InnerState, transcript: Transcript
    ) -> tuple[_InnerState, Transcript, RoundMsg]:
        return _jagged_assist_step(state, transcript)


def _merged_bits(col_prefix_sums: Array, cfg: JaggedStaticConfig) -> Array:
    """(l_max, 2·n_d) merged prefix bits: row c = bits(t_c) ‖ bits(t_{c+1})."""
    left = col_prefix_sums[: cfg.l_max]  # (l_max, n_d)
    right = col_prefix_sums[1:]  # (l_max, n_d)
    return jnp.concatenate([left, right], axis=1)


@partial(register_dataclass, data_fields=["round_polys", "point"], meta_fields=[])
@dataclass(frozen=True)
class JaggedAssistProof:
    """Inner-sumcheck proof.

    round_polys: (2·n_d, 3) stacked degree-2 monomial coefficients.
    point: (2·n_d,) the reduced challenge point in buffer (MSB-first) order —
        challenges[::-1] of sampling order, so the verifier reverses back.

    A registered pytree so the open/verify cores cross the `@jit` boundary as a
    single leaf-bearing container."""

    round_polys: Array
    point: Array


@partial(jax.jit, static_argnames=("cfg",))
def _prove_setup(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    transcript: Transcript,
    *,
    cfg: JaggedStaticConfig,
) -> tuple[_InnerState, Array, Transcript]:
    """Compute the claim, seed the transcript, and build the round-0 carry — one
    `@jit` zone (the claim eval + one sponge absorb) ahead of the per-round loop."""
    dtype = z_row.dtype
    claimed_sum = eval_jagged_mle(col_prefix_sums, z_row, z_col, z_final, cfg=cfg)
    transcript = transcript.observe(claimed_sum)
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones([], dtype=dtype))  # 2^n_c
    merged_bits = _merged_bits(col_prefix_sums, cfg)
    state = _InnerState(
        weights=col_eq[: cfg.l_max],
        buf=merged_bits,
        merged_bits=merged_bits,
        claim=claimed_sum,
        rounds_left=jnp.int32(2 * cfg.n_d),
        z_row=z_row,
        z_final=z_final,
        t_matrix=jnp.asarray(_TRANSITION_ROWS, dtype=dtype),
        num_vars=max(cfg.n_r, cfg.n_d),
    )
    return state, claimed_sum, transcript


def _prove_core(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    transcript: Transcript,
    *,
    cfg: JaggedStaticConfig,
) -> tuple[JaggedAssistProof, Array, Transcript]:
    """`prove_jagged_eval` body: jitted setup, then the `fold_rounds` Python loop
    over the jitted per-round step. The round body compiles once (round-invariant
    carry shapes) and is reused across all 2·n_d rounds — unrolling all rounds into
    a single `@jit` instead re-expands the sponge permute per round, a multi-minute
    XLA compile on the ZKX CPU backend."""
    state, claimed_sum, transcript = _prove_setup(
        col_prefix_sums, z_row, z_col, z_final, transcript, cfg=cfg
    )
    _, transcript, msgs = fold_rounds(
        JaggedAssistRound(), state, transcript, 2 * cfg.n_d
    )
    round_polys = jnp.stack([m.round_poly for m in msgs])  # (n_vars, 3)
    # Sampling order is LSB→MSB of the buffer; reverse to buffer (MSB-first) order
    # so the verifier's reversed point splits cleanly into (z_left, z_right).
    point = jnp.stack([m.challenge for m in msgs])[::-1]
    proof = JaggedAssistProof(round_polys=round_polys, point=point)
    return proof, claimed_sum, transcript


def prove_jagged_eval(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    *,
    cfg: JaggedStaticConfig,
    transcript: Transcript,
) -> tuple[JaggedAssistProof, Array, Transcript]:
    """Open the inner jagged-assist sumcheck.

    Returns (proof, claimed_sum, transcript) where claimed_sum = J̃(z_row, z_col,
    z_final) = eval_jagged_mle(...).
    """
    return _prove_core(col_prefix_sums, z_row, z_col, z_final, transcript, cfg=cfg)


@jax.jit
def _verify_round_step(
    coef: Array, claim: Array, transcript: Transcript
) -> tuple[Array, Array, Transcript, Array]:
    """One jitted verifier replay round: check claim == s(0)+s(1), observe the
    coefs, sample α, reduce claim to s(α). Returns (new_claim, α, transcript, ok).
    Compiles once (round-invariant shapes) and is reused across rounds."""
    c0, c1, c2 = coef[0], coef[1], coef[2]
    ok = claim == c0 + (c0 + c1 + c2)  # s(0) + s(1)
    # Observe + sample IDENTICALLY to the prover round so the transcripts replay
    # in lockstep: one base-field challenge cast to the EF's canonical base, then
    # folded into the EF claim directly.
    transcript, r = transcript.observe_and_sample(coef, 1)
    alpha = r[0].astype(zk_dtypes.efinfo(coef.dtype).base_field_dtype)
    new_claim = c0 + alpha * (c1 + alpha * c2)  # s(α)
    return new_claim, alpha, transcript, ok


@partial(jax.jit, static_argnames=("cfg",))
def _verify_seed(
    proof: JaggedAssistProof, transcript: Transcript, *, cfg: JaggedStaticConfig
) -> tuple[Array, Transcript]:
    """Reconstruct the opened value from round 0 and seed the transcript with it.

    The prover observed claimed_sum; the first round's c0 + (c0+c1+c2) == s(0)+s(1)
    == claimed_sum by construction, so it doubles as the seed and the return."""
    jagged_eval = proof.round_polys[0, 0] + (
        proof.round_polys[0, 0] + proof.round_polys[0, 1] + proof.round_polys[0, 2]
    )
    return jagged_eval, transcript.observe(jagged_eval)


@partial(jax.jit, static_argnames=("cfg",))
def _verify_final_check(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    final_point: Array,
    final_eval: Array,
    *,
    cfg: JaggedStaticConfig,
) -> Array:
    """The reduced branching-program leaf check: split final_point into
    (z_left, z_right), recompute eq_sum·h_bp, and test against final_eval."""
    dtype = z_row.dtype
    z_left = final_point[: cfg.n_d]
    z_right = final_point[cfg.n_d :]

    # eq_weighted(final_point) =
    #   Σ_c eq(z_col, c)·eval_eq(t_c ‖ t_{c+1}, final_point).
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones([], dtype=dtype))
    merged_bits = _merged_bits(col_prefix_sums, cfg)
    eq_sum = _ef_sum(
        [col_eq[c] * eval_eq(merged_bits[c], final_point) for c in range(cfg.l_max)]
    )

    # h_bp(final_point) — the single branching-program leaf the verifier computes.
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    h_bp = bp_eval_core(
        z_row, z_final, z_left, z_right, t_matrix, max(cfg.n_r, cfg.n_d)
    )
    return eq_sum * h_bp == final_eval


def _verify_core(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    proof: JaggedAssistProof,
    transcript: Transcript,
    *,
    cfg: JaggedStaticConfig,
) -> tuple[Array, Array, Transcript]:
    """`verify_jagged_eval` body: jitted seed, a Python replay loop over the jitted
    per-round step (the per-variable verifier returns its sampled challenge, so it
    is a local loop, not VerifyChain), then the jitted final leaf check. Per-round
    jit keeps the sponge permute from re-unrolling per round (see `_prove_core`)."""
    n_vars = 2 * cfg.n_d
    jagged_eval, transcript = _verify_seed(proof, transcript, cfg=cfg)

    claim = jagged_eval
    ok = jnp.array(True)
    challenges: list[Array] = []
    for r in range(n_vars):
        claim, alpha, transcript, ok_round = _verify_round_step(
            proof.round_polys[r], claim, transcript
        )
        challenges.append(alpha)
        ok = ok & ok_round

    # Reverse sampling order → buffer (MSB-first) order, split (z_left, z_right).
    final_point = jnp.stack(challenges)[::-1]
    ok = ok & _verify_final_check(
        col_prefix_sums, z_row, z_col, z_final, final_point, claim, cfg=cfg
    )
    return jagged_eval, ok, transcript


def verify_jagged_eval(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    proof: JaggedAssistProof,
    *,
    cfg: JaggedStaticConfig,
    transcript: Transcript,
) -> tuple[Array, Array, Transcript]:
    """Verify the inner jagged-assist sumcheck.

    Returns (jagged_eval, ok, transcript) where jagged_eval is the proven
    J̃(z_row, z_col, z_final) and ok ANDs every per-round check with the final
    branching-program leaf consistency check.
    """
    return _verify_core(
        col_prefix_sums, z_row, z_col, z_final, proof, transcript, cfg=cfg
    )
