# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged PCS prover: commit + the S1-S6 opening seam.

`JaggedPcsProver` owns both halves of the prover:

* `commit` packs variable-height blocks into the dense stacked MLE `D`, commits
  it via the injected multilinear `PcsProver` (BaseFold), binds the jagged
  structure into the root, and retains `JaggedProverData` (the BaseFold witness,
  `D`, the layout, and the per-column prefix-sum bit tensor) for `open`.
* `open` runs the four-step opening that ties the prior pieces together — sample
  `z_col`, materialize the jagged indicator `J̃` (S2), run the outer Hadamard
  sumcheck `Σ_i D(i)·J̃(i) = claim` (S3), reprove `J̃(z_row, z_col, z_final)` via
  the inner jagged-assist sumcheck (S4), and open `D(z_final)` via the stacked
  BaseFold opening (S6) — all in one Fiat-Shamir transcript so the dual verifier
  replays in lockstep.

The seam requires the dense MLE and the jagged indicator to share a variable
count: `layout.log_m == cfg.n_d`. The dense buffer `D` is `2^log_m`; the
indicator `J̃` from `partial_eval` is `2^n_d`; the outer sumcheck Hadamard-pairs
them index-for-index (the `from_blocks` column-major packing puts column `c`'s
row `r` at flat index `t_c + r`, exactly where `J̃` is nonzero), so they must be
the same length. `from_blocks` derives its tier with the same `log_area_tier`
the indicator uses, so the match holds by construction for any
`log_stacking_height <= n_d` (`K = 2^(n_d - log_s)`); `open` still asserts it
to reject an over-tall stacking height.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_dataclass

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.jagged.config import JaggedOpeningProof
from zorch.pcs.jagged.dense import JaggedLayout, from_blocks
from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    JaggedStaticConfig,
    _offset_bit_tensor,
    bp_eval_core,
    build_jagged_layout,
    eval_jagged_mle,
    partial_eval,
)
from zorch.pcs.jagged.stacked import stacked_open
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import eval_univariate
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.prover import RoundMsg, SumcheckRound, prove
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize

# Outer Hadamard sumcheck degree: Σ_i D(i)·J̃(i), a product of two MLEs.
_OUTER_DEGREE = 2

# Inner jagged-assist sumcheck degree (eq-weighted branching-program summand).
_INNER_DEGREE = 2


def _ef_sum(terms: list[Array]) -> Array:
    """Σ over a static-length list of extension-field scalars at trace time.

    jnp.sum over an extension-field array aborts the ZKX backend, so the column
    reduction is a Python-level fold over the static l_max (same semantics)."""
    return functools.reduce(operator.add, terms)


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

    # zorch's native round message is eval form [s(0), s(1), s(2)], checked by the
    # stock `SumcheckRound` (ok = claim == s(0)+s(1), reduce via eval_univariate).
    # From the monomial coefs c0 = p_0, c1 = claim − 2·p_0 − p_inf, c2 = p_inf:
    #   s(0) = c0 = p_0
    #   s(1) = c0 + c1 + c2 = claim − p_0
    #   s(2) = c0 + 2·c1 + 4·c2 = 2·claim − 3·p_0 + 2·p_inf
    s0 = p_0
    s1 = state.claim - p_0
    s2 = 2 * state.claim - 3 * p_0 + 2 * p_inf
    msg = jnp.stack([s0, s1, s2])

    # Observe + sample IDENTICALLY to `SumcheckRound` so prover and verifier
    # transcripts stay aligned. The sponge squeezes a Montgomery base scalar that
    # multiplies the Montgomery EF state directly (all-Montgomery world, no cast).
    transcript, r = transcript.observe_and_sample(msg, 1)
    alpha = r[0]

    # Fold: bind this column to α and update the running weights/claim. Reduce the
    # claim via the stock eval-form Lagrange interpolation (eval_univariate).
    new_buf = state.buf.at[:, round_idx].set(jnp.broadcast_to(alpha, col_shape))
    new_weights = state.weights * (alpha * bits_i + (one - alpha) * (one - bits_i))
    new_claim = eval_univariate(msg, alpha)

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
    return new_state, transcript, RoundMsg(msg, alpha)


class JaggedAssistRound(Round):
    """One degree-2 jagged-assist sumcheck round (prover side).

    Maps (state, transcript) -> (state, transcript, RoundMsg). Per round:
    eliminate the next merged-bit variable by evaluating the branching program at
    the column's 0/1 settings over the L deltas, build the eval-form round poly
    [s(0), s(1), s(2)], observe + sample alpha, then fold — bind that column to
    alpha, update the running weights by eq(alpha, bit_c), and reduce the claim to
    s(alpha). The fold lands in the emitted state, so the round is self-contained
    and `fold_rounds` chains it straight through with no prior-challenge plumbing.
    Delegates to the jitted `_jagged_assist_step` so the round body compiles once
    and is reused across rounds."""

    def __call__(
        self, state: _InnerState, transcript: Transcript
    ) -> tuple[_InnerState, Transcript, RoundMsg]:
        return _jagged_assist_step(state, transcript)


def _merged_bits(col_prefix_sums: Array, cfg: JaggedStaticConfig) -> Array:
    """(l_max, 2·n_d) merged prefix bits: row c = bits(t_c) ‖ bits(t_{c+1})."""
    left = col_prefix_sums[: cfg.l_max]  # (l_max, n_d)
    right = col_prefix_sums[1:]  # (l_max, n_d)
    return jnp.concatenate([left, right], axis=1)


@partial(jax.jit, static_argnames=("cfg",))
def _inner_setup(
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


def prove_jagged_eval(
    col_prefix_sums: Array,
    z_row: Array,
    z_col: Array,
    z_final: Array,
    *,
    cfg: JaggedStaticConfig,
    transcript: Transcript,
) -> tuple[Array, Array, Transcript]:
    """Open the inner jagged-assist sumcheck reproving J̃(z_row, z_col, z_final).

    Returns (inner_round_polys, claimed_sum, transcript) where claimed_sum =
    J̃(z_row, z_col, z_final) = eval_jagged_mle(...) and `inner_round_polys` is the
    `(2·n_d, 3)` eval-form proof the stock `verify(SumcheckRound(2))` replays.

    Jitted setup (claim eval + one sponge absorb), then the `fold_rounds` Python
    loop over the jitted per-round step. The round body compiles once
    (round-invariant carry shapes) and is reused across all 2·n_d rounds —
    unrolling all rounds into a single `@jit` instead re-expands the sponge permute
    per round, a multi-minute XLA compile on the ZKX CPU backend.
    """
    state, claimed_sum, transcript = _inner_setup(
        col_prefix_sums, z_row, z_col, z_final, transcript, cfg=cfg
    )
    _, transcript, msgs = fold_rounds(
        JaggedAssistRound(), state, transcript, 2 * cfg.n_d
    )
    inner_round_polys = jnp.stack([m.round_poly for m in msgs])  # (2·n_d, 3)
    return inner_round_polys, claimed_sum, transcript


def _expand_col_heights(layout: JaggedLayout) -> list[int]:
    """Per-column heights for the indicator: each block `[h, w]` is `w` unit-width
    columns of height `h` (matching `from_blocks`' column-major packing)."""
    return [h for h, w in zip(layout.heights, layout.widths) for _ in range(w)]


def _indicator_inputs(
    layout: JaggedLayout,
) -> tuple[Array, Array, JaggedStaticConfig]:
    """`(col_prefix_sums, col_prefix_sums_offsets, cfg)` for a committed region.

    `col_prefix_sums` is the field-valued bit tensor `build_jagged_layout`
    produces (correct for the inner sumcheck + `eval_jagged_mle` arithmetic);
    `col_prefix_sums_offsets` is the canonical-limb sibling `partial_eval`'s raw
    offset bitcast needs (see `_offset_bit_tensor`). `l_max` is the exact column
    count `L = Σ widths`; `n_r` covers the tallest column. The
    `cfg.n_d == layout.log_m` requirement is enforced in `open` (commit itself
    does not need it).
    """
    col_heights = _expand_col_heights(layout)
    l_max = max(len(col_heights), 1)
    n_r = max(log2_ceil_usize(max(col_heights, default=1)), 1)
    col_prefix_sums, cfg = build_jagged_layout(
        col_heights, l_max=l_max, n_r=n_r, dtype=layout.dtype
    )
    offsets = _offset_bit_tensor(col_heights, l_max, cfg)
    return col_prefix_sums, offsets, cfg


@partial(
    register_dataclass,
    data_fields=[
        "basefold_prover_data",
        "dense",
        "col_prefix_sums",
        "col_prefix_sums_offsets",
    ],
    meta_fields=["layout", "cfg"],
)
@dataclass(frozen=True)
class JaggedProverData:
    """Retained witness from `JaggedPcsProver.commit`, everything `open` needs.

    basefold_prover_data: the underlying BaseFold witness over the dense `[S, K]`
        matrix (the stacked opening reads it).
    dense: the dense MLE `D`, shape `(2^log_m,)` (= `(2^n_d,)`); the outer
        sumcheck Hadamard-pairs it with the indicator and the stacked opening
        proves `D(z_final)`.
    col_prefix_sums: the `(l_max+1, n_d)` field-valued prefix-sum bit tensor (the
        S4 inner sumcheck + `eval_jagged_mle` read it as field elements).
    col_prefix_sums_offsets: the canonical-limb sibling for `partial_eval`'s raw
        offset bitcast (S2) — see `_offset_bit_tensor`.
    layout / cfg: static structure metadata (jagged layout + indicator config).

    A registered pytree (layout/cfg static) so it threads a `@jit` boundary.
    """

    basefold_prover_data: BasefoldProverData
    dense: Array
    col_prefix_sums: Array
    col_prefix_sums_offsets: Array
    layout: JaggedLayout
    cfg: JaggedStaticConfig


class JaggedPcsProver:
    """Commits variable-height blocks via an injected multilinear `PcsProver` and
    opens them at a row point, binding the jagged structure into the commitment.

    `sponge` / `compressor` perform the structure bind (the jagged layer owns it,
    not the PCS). The jit'd device commit zone is built once in `__init__`,
    capturing `prover` / `sponge` / `compressor`, keyed on the MLE + structure
    shapes, so it compiles once per (area tier, block count).
    """

    _commit: Any  # jax.jit-wrapped device-commit closure, built in __init__

    def __init__(
        self, prover: BasefoldProver, sponge: Sponge, compressor: Compression
    ) -> None:
        if sponge.out != compressor.chunk:
            raise ValueError(
                f"sponge digest ({sponge.out}) must equal compressor chunk "
                f"({compressor.chunk}) for the structure bind"
            )
        if compressor.arity != 2:
            raise ValueError(
                f"compressor.arity must be 2 for the (root, structure_hash) "
                f"bind, got {compressor.arity}"
            )
        self.prover = prover
        self.sponge = sponge
        self.compressor = compressor

        @jax.jit
        def _commit(mle: Array, structure: Array) -> tuple[Array, Any]:
            # The seam commits a Sequence of column MLEs; BaseFold binds them
            # jointly under one root (a matrix commitment), which the structure
            # bind below hashes against.
            columns = [mle[:, j] for j in range(mle.shape[1])]
            root, prover_data = prover.commit(columns)
            structure_hash = sponge.hash(structure)
            bound = compressor.compress(jnp.stack([root, structure_hash]))
            return bound, prover_data

        self._commit = _commit

    @property
    def bf_prover(self) -> BasefoldProver:
        return self.prover

    def commit(
        self, blocks: list[Array], *, log_stacking_height: int
    ) -> tuple[Array, JaggedProverData]:
        """Pack + commit; return `(commitment, JaggedProverData)`.

        `from_blocks` (host) derives the tier and the stacked MLE `[S, K]`; the
        `@jit` zone commits it via `pcs` and binds the structure hash into the
        root. The retained `JaggedProverData` carries the dense buffer, the
        layout, and the per-column prefix-sum bit tensor for `open`.
        """
        packed, layout = from_blocks(blocks, log_stacking_height=log_stacking_height)
        mle = packed.reshape(layout.K, layout.S).T  # [S, K]
        structure = _structure_vec(layout)
        col_prefix_sums, offsets, cfg = _indicator_inputs(layout)
        bound, basefold_prover_data = self._commit(mle, structure)
        prover_data = JaggedProverData(
            basefold_prover_data=basefold_prover_data,
            dense=packed,
            col_prefix_sums=col_prefix_sums,
            col_prefix_sums_offsets=offsets,
            layout=layout,
            cfg=cfg,
        )
        return bound, prover_data

    def open(
        self,
        prover_data: JaggedProverData,
        z_row: Array,
        column_claims: Array,
        transcript: Transcript,
    ) -> tuple[JaggedOpeningProof, Transcript]:
        """Open the committed region at `z_row`.

        `column_claims` is the `(L,)` vector of per-column claims
        `{MLE_c(z_row)}` (one per unit-width column, `L = Σ widths`), in the
        flat column order `from_blocks` packs — passed for transcript symmetry
        with `verify` (it samples `z_col` then folds the claim); zorch's stock
        sumcheck is seed-free, so the prover derives the round polys straight
        from `D`·`J̃` and never reads `column_claims`. Returns
        `(proof, transcript)`.
        """
        cfg = prover_data.cfg
        layout = prover_data.layout
        col_prefix_sums = prover_data.col_prefix_sums
        if z_row.shape != (cfg.n_r,):
            raise ValueError(
                f"z_row must have shape (n_r,)=({cfg.n_r},), got {z_row.shape}"
            )
        if column_claims.shape[0] > cfg.l_max:
            raise ValueError(
                f"column_claims holds at most l_max={cfg.l_max} per-column claims, "
                f"got {column_claims.shape[0]}"
            )
        del column_claims  # symmetry with verify; the seed-free sumcheck needs it not
        if cfg.n_d != layout.log_m:
            raise ValueError(
                f"jagged open requires cfg.n_d == layout.log_m; got n_d={cfg.n_d}, "
                f"log_m={layout.log_m}. Choose log_stacking_height <= n_d (the "
                f"log-area tier) so the dense MLE and the jagged indicator share "
                f"a variable count."
            )

        # S1: sample z_col. The transcript squeezes base-field scalars; embed them
        # in the indicator's extension field so the jagged arithmetic stays
        # single-field (base·EF mixed-stacks fail to promote inside the branching
        # program). z_col must be sampled here (not in verify alone) so both
        # transcripts advance identically before the outer sumcheck.
        transcript, z_col = transcript.sample(cfg.n_c)
        z_col = z_col.astype(cfg.dtype)

        # S2: materialize the jagged indicator J̃(z_row, z_col, ·). `partial_eval`
        # reads its scatter offsets from the canonical-limb tensor.
        indicator = partial_eval(
            prover_data.col_prefix_sums_offsets, z_row, z_col, cfg=cfg
        )

        # S3: outer Hadamard sumcheck Σ_i D(i)·J̃(i) = claim.
        folded, transcript, msgs = prove(
            SumcheckRound(_OUTER_DEGREE), [prover_data.dense, indicator], transcript
        )
        outer_sumcheck_polys = msgs.round_poly  # [n_d, degree+1]
        # zorch's stock sumcheck folds MSB-first (buf[:half] = first variable),
        # so the challenge stream is already MSB→LSB — the order `eval_mle` /
        # the stacked opening / the inner BP all read. (The whir-zorch reference
        # folds LSB-first with insert-at-front and reverses; zorch does not.)
        z_final = msgs.challenge.astype(cfg.dtype)
        outer_final_eval = folded[0][0] * folded[1][0]  # D(z_final)·J̃(z_final)

        # S4: inner jagged-assist sumcheck reproving J̃(z_row, z_col, z_final).
        inner_round_polys, inner_claimed_sum, transcript = prove_jagged_eval(
            col_prefix_sums, z_row, z_col, z_final, cfg=cfg, transcript=transcript
        )

        # S6: stacked BaseFold opening of D(z_final). Its `dense_eval` (recomputed
        # from the column evals) is the one the wire carries — the stacked verify
        # re-checks it against the opened columns, and the S5 product check ties it
        # back to the outer sumcheck's reduced D(z_final), so one source suffices.
        dense_eval, column_values, basefold_proof, transcript = stacked_open(
            self.bf_prover,
            prover_data.basefold_prover_data,
            z_final,
            layout,
            transcript,
        )

        # The un-bound BaseFold root the jagged commitment binds; the verifier
        # re-derives the bind from it and feeds it to the stacked verify.
        basefold_root = prover_data.basefold_prover_data.digest_layers[-1][0]

        proof = JaggedOpeningProof(
            outer_sumcheck_polys=outer_sumcheck_polys,
            outer_final_eval=outer_final_eval,
            inner_round_polys=inner_round_polys,
            inner_claimed_sum=inner_claimed_sum,
            dense_eval=dense_eval,
            column_values=column_values,
            basefold_root=basefold_root,
            basefold_proof=basefold_proof,
        )
        return proof, transcript


def _structure_vec(layout: JaggedLayout) -> Array:
    """[num_blocks, heights..., widths...] as a base-field row, for binding.

    Counts are bounded by M_max = 2^log_m < p for all supported field primes,
    so they round-trip through the field dtype without modular reduction.
    """
    vals = [len(layout.heights), *layout.heights, *layout.widths]
    return jnp.asarray(vals, dtype=layout.dtype)


def _compress_column_claims(column_claims: Array, z_col: Array) -> Array:
    """claim = Σ_c eq(z_col, c)·column_claims[c].

    `column_claims` holds `L = Σ widths` real per-column claims; the eq weights
    span the padded `2^n_c` column hypercube, so take the first `L` weights.
    EF sum via a trace-time fold (never `jnp.sum` over EF — ZKX abort)."""
    weights = expand_eq_to_hypercube(z_col, jnp.ones((), z_col.dtype))  # (2^n_c,)
    L = column_claims.shape[0]
    terms = [weights[c] * column_claims[c] for c in range(L)]
    return functools.reduce(operator.add, terms)
