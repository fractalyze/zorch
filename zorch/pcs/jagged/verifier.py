# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 jagged-eval verifier: the zorch-native dual of the stage-5 prover.

``verify_jagged_eval_msg`` replays the sumcheck half (``JaggedEvalRound``):
the outer Hadamard sumcheck against the column claim recomputed from the
zerocheck openings, the inner branching-program sumcheck against the proof's
claimed J̃, the succinct branching-program leaf check at the reduced point,
and the product check tying ``D(z_final)·J̃`` to the outer reduction.
``stacked_basefold_verify`` is the dual of ``stacked_basefold_open``'s SP1
wire: the stacking check binding ``D(z_final)`` to the per-round batch
evaluations, the interleaved-sumcheck round identities, the proof-of-work
check, and the query phase — component openings against the carry's
separator-bound commitments, their staggered RLC tied to fold layer 0, and
the per-layer fold chain down to the constant final poly.

Both run on zorch verifier blocks (``CoeffsSumcheckRound`` under the
``zorch.verify`` scan, ``bp_eval_core``, ``verify_fold_chain``); what stays
here is the SP1 wire — the absorb schedule, the canonical-low-bits query
rule, the separator-bound fold-layer roots, and the scalar final poly.

References (pinned at the same SP1 commit as ``zerocheck/jagged.py``):
- jagged verify — ``verify_trusted_evaluations``:
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/jagged/src/verifier.rs
- inner leaf check — ``jagged_evaluation``:
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/jagged/src/jagged_eval/sumcheck_eval.rs
- stacking check — ``verify_trusted_evaluation``:
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/stacked/src/verifier.rs
- fold/query phase — ``verify_mle_evaluations``:
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/basefold/src/verifier.rs
  SP1 checks each query's fold chain against the final poly and never
  separately compares the sumcheck side's terminal claim to it; the dual
  mirrors SP1's check set exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from zk_dtypes import efinfo

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.merkle import Opening
from zorch.commit.smcs import SingleMatrixCommitmentScheme, VerifyCode
from zorch.pcs.basefold.batching import batch_staggered
from zorch.pcs.fold import verify_fold_chain
from zorch.pcs.jagged.open import (
    StackedOpenProof,
    sample_query_positions,
    sample_rlc_coeffs,
)
from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    bp_eval_core,
    build_jagged_layout,
)
from zorch.pcs.jagged.prover import (
    JaggedEvalMsg,
    merged_prefix_bits,
    outer_sumcheck_claim,
)
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge
from zorch.utils.bits import log2_strict_usize
from zorch.verify import verify

# Outer Hadamard / inner branching-program sumcheck degree (matches the
# prover's coefficient-form round polys).
_DEGREE = 2


def verify_jagged_eval_msg(
    col_heights: Sequence[int],
    all_claims: Array,
    z_row: Array,
    z_col: Array,
    msg: JaggedEvalMsg,
    transcript: Transcript,
    *,
    dtype: Any,
) -> tuple[Transcript, Array, Array]:
    """Verify the sumcheck half of the stage-5 proof; returns
    ``(transcript, z_final, ok)``.

    ``col_heights`` / ``all_claims`` are the statement-side column manifest
    and the per-column claims assembled from the zerocheck opened values
    (``assemble_columns``); ``z_row`` is the zerocheck point in SP1's
    insert-at-front order and ``z_col`` the caller's own column samples.
    ``z_final`` is the dual's own outer reduction point (pinned against the
    wire copy), which the stacked open consumes. ``ok`` ANDs every
    acceptance leg; malformed proof *structure* raises instead — shapes are
    host decisions, the same split as the zerocheck dual.
    """
    ef_limbs = efinfo(dtype).degree
    heights = list(col_heights)
    l_max = len(heights)
    _, cfg = build_jagged_layout(heights, l_max, z_row.shape[0], dtype)
    num_bits = cfg.n_d
    if msg.inner_sumcheck_polys.shape != (2 * num_bits, _DEGREE + 1):
        raise ValueError(
            f"need one degree-{_DEGREE} coefficient poly per merged prefix "
            f"bit ({2 * num_bits}, {_DEGREE + 1}), got "
            f"{msg.inner_sumcheck_polys.shape}"
        )

    # The outer claim is the verifier's own derivation from the leaf-checked
    # claims; the wire copy is pinned so a stale serialization cannot drift.
    claim = outer_sumcheck_claim(all_claims, z_col)
    ok = jnp.array_equal(claim, msg.outer_sumcheck_claim)

    point, outer_final, transcript, ok_rounds = verify(
        CoeffsSumcheckRound(_DEGREE, ef_limbs),
        claim,
        msg.outer_sumcheck_polys,
        transcript,
    )
    z_final = point[::-1]
    ok = ok & ok_rounds & jnp.array_equal(z_final, msg.outer_sumcheck_point)

    # SP1 absorbs the claimed J̃ value before the inner rounds
    # (fractalyze/sp1-zorch#90), then replays them with the same rule.
    transcript = transcript.observe(msg.inner_claimed_sum)
    ipoint, inner_final, transcript, ok_inner = verify(
        CoeffsSumcheckRound(_DEGREE, ef_limbs),
        msg.inner_claimed_sum,
        msg.inner_sumcheck_polys,
        transcript,
    )
    inner_point = ipoint[::-1]
    ok = ok & ok_inner & jnp.array_equal(inner_point, msg.inner_point)

    # The succinct leaf check: at the bound buffer point, the prover's
    # eq-weighted branching-program sum collapses to one bp evaluation
    # (every column's bound row is the same challenge vector) weighted by
    # Σ_c eq(z_col, c)·eq(merged_c, inner_point).
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)
    merged = merged_prefix_bits(heights, num_bits, dtype=dtype)
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))
    eqs = jax.vmap(lambda row: eval_eq(row, inner_point))(merged)
    h_bp = bp_eval_core(
        z_row,
        z_final,
        inner_point[:num_bits],
        inner_point[num_bits:],
        t_matrix,
        max(cfg.n_r, cfg.n_d),
    )
    ok = ok & jnp.array_equal(inner_final, jnp.sum(col_eq[:l_max] * eqs) * h_bp)

    # Product check: the outer reduction's final claim is D(z_final)·J̃,
    # with J̃ the inner sumcheck's (now verified) claimed sum and D(z_final)
    # the wire value the stacked open binds and opens.
    ok = ok & jnp.array_equal(msg.dense_eval * msg.inner_claimed_sum, outer_final)
    return transcript, z_final, ok


def _ef_pairs(rows: Array, dtype: Any) -> Array:
    """The ``(Q, 2·limbs)`` base-field pair-leaf rows as ``(Q, 2)`` extension
    values — the inverse of the open's leaf bitcast."""
    limbs = efinfo(dtype).degree
    return jax.lax.bitcast_convert_type(rows.reshape(rows.shape[0], 2, limbs), dtype)


def stacked_basefold_verify(
    smcs: SingleMatrixCommitmentScheme,
    code: BitReversedReedSolomon,
    round_widths: Sequence[int],
    z_final: Array,
    dense_eval: Array,
    log_stacking_height: int,
    proof: StackedOpenProof,
    transcript: GrindingTranscript,
    *,
    num_queries: int,
    pow_bits: int,
) -> tuple[GrindingTranscript, Array]:
    """Verify a stacked BaseFold open; returns ``(transcript, ok)``.

    ``round_widths`` is the statement-side stacked column count of each
    round (its aligned area over the stacking height). Openings verify
    against the proof's own ``component_commitments`` — soundness comes
    from the caller tying each of those to the statement's commitment via
    the structure rebind (the jagged level's check, as in SP1). The
    transcript must be the grinding transcript at the state SP1 enters the
    open with, exactly as on the prover side.
    """
    ef = dense_eval.dtype
    bf = code.dtype
    ef_limbs = efinfo(ef).degree
    num_vars = log_stacking_height
    block_len = code.block_len

    if num_vars < 1:
        # SP1 fixes a positive stacking height, so a zero-variable open (no
        # fold layers, no query openings) has no SP1 check set to mirror —
        # reject it up front rather than index-error on the empty query phase.
        raise ValueError(
            f"need at least one stacking variable, got "
            f"log_stacking_height={log_stacking_height}"
        )
    if not (
        len(proof.component_commitments)
        == len(round_widths)
        == len(proof.batch_evals)
        == len(proof.component_openings)
    ):
        raise ValueError(
            f"round mismatch: {len(proof.component_commitments)} component "
            f"commitments, {len(round_widths)} widths, "
            f"{len(proof.batch_evals)} batch evals, "
            f"{len(proof.component_openings)} component openings"
        )
    for r, (evals, k) in enumerate(zip(proof.batch_evals, round_widths)):
        if evals.shape != (k,):
            raise ValueError(
                f"round {r} batch evals must cover its {k} stacked columns, "
                f"got shape {evals.shape}"
            )
    if proof.univariate_messages.shape != (num_vars, 2):
        raise ValueError(
            f"need one (s(0), s(1)) message per stacking variable "
            f"({num_vars}, 2), got {proof.univariate_messages.shape}"
        )
    if (
        proof.fri_commitments.shape[0] != num_vars
        or proof.fri_raw_roots.shape[0] != num_vars
        or len(proof.query_openings) != num_vars
    ):
        raise ValueError(
            f"need one committed fold layer per stacking variable "
            f"({num_vars}), got {proof.fri_commitments.shape[0]} bound roots / "
            f"{proof.fri_raw_roots.shape[0]} raw roots / "
            f"{len(proof.query_openings)} query openings"
        )
    if z_final.shape[0] < num_vars:
        raise ValueError(
            f"the folded point must carry at least the {num_vars} stacking "
            f"variables, got {z_final.shape[0]}"
        )

    stack_point = z_final[-num_vars:]
    batch_point = z_final[: z_final.shape[0] - num_vars]

    # SP1's stacking check: the claimed D(z_final) interpolates the batch
    # evaluations at the leading (column-selecting) coordinates.
    flat = jnp.concatenate(list(proof.batch_evals))
    width = 1 << batch_point.shape[0]
    if flat.shape[0] > width:
        raise ValueError(
            f"{flat.shape[0]} stacked columns do not fit the "
            f"{batch_point.shape[0]} leading point variables"
        )
    padded = jnp.concatenate([flat, jnp.zeros((width - flat.shape[0],), ef)])
    ok = jnp.array_equal(dense_eval, eval_mle(padded, batch_point))

    # The open's absorb schedule: the scalar D(z_final), then each round's
    # batch evaluations (the component roots were bound upstream at commit).
    t: GrindingTranscript = transcript
    t = t.observe(dense_eval)
    for evals in proof.batch_evals:
        t = t.observe(evals)

    total_width = sum(int(k) for k in round_widths)
    t, coeffs = sample_rlc_coeffs(t, total_width, ef)
    claim = batch_staggered(list(proof.batch_evals), coeffs)

    t = t.observe(jnp.asarray(num_vars, bf))

    # Interleaved sumcheck replay: round i binds stack_point's last unbound
    # variable, so the claim identity is claim == (1-x)·s(0) + x·s(1) and the
    # fold reduction s(0) + β·s(1). Each bound fold-layer root is observed
    # before β; the raw root is the serializer's wire copy, pinned through
    # the shared separator rebind.
    one = jnp.ones((), ef)
    betas: list[Array] = []
    for i in range(num_vars):
        m = proof.univariate_messages[i]
        x = stack_point[num_vars - 1 - i]
        ok = ok & jnp.array_equal(claim, (one - x) * m[0] + x * m[1])
        t = t.observe(m)
        t = t.observe(proof.fri_commitments[i])
        t, beta = sample_challenge(t, ef, ef_limbs)
        betas.append(beta)
        claim = m[0] + beta * m[1]
        log_h = log2_strict_usize(block_len >> (i + 1))
        ok = ok & jnp.array_equal(
            smcs.bind_root(proof.fri_raw_roots[i], log_h, 2 * ef_limbs, bf),
            proof.fri_commitments[i],
        )

    t = t.observe(jnp.atleast_1d(proof.final_poly))
    t, ok_pow = t.check_witness(pow_bits, proof.pow_witness)
    ok = ok & ok_pow

    t, positions = sample_query_positions(t, block_len, num_queries)
    layer_pos = code.layer_positions(positions, num_vars)

    # Merkle phase: every component matrix's opened rows rebuild its bound
    # commitment at the query positions, every fold layer's pair-leaves
    # rebuild its bound root at the halved index.
    def verify_rows(
        root: Array,
        dims: tuple[int, int],
        idx: Array,
        rows: Array,
        paths: list[Array],
    ) -> Array:
        codes = jax.vmap(
            lambda i, row, path: smcs.verify_batch(root, dims, i, row, path)
        )(idx, rows, paths)
        return jnp.all(codes == int(VerifyCode.OK))

    for r, (rows, paths) in enumerate(proof.component_openings):
        ok = ok & verify_rows(
            proof.component_commitments[r],
            (block_len, int(round_widths[r])),
            positions,
            rows,
            paths,
        )
    for i, (rows, paths) in enumerate(proof.query_openings):
        ok = ok & verify_rows(
            proof.fri_commitments[i],
            (block_len >> (i + 1), 2 * ef_limbs),
            layer_pos[i],
            rows,
            paths,
        )

    # Fold consistency: the staggered RLC of the component rows is the
    # batched codeword at `positions` — the matching leg of fold layer 0's
    # opened pair — and each layer's pair folds to the next layer's opened
    # value, down to the constant final poly.
    query_ops = [
        Opening(row=_ef_pairs(rows, ef), path=paths)
        for rows, paths in proof.query_openings
    ]
    comp_val = batch_staggered([rows for rows, _ in proof.component_openings], coeffs)
    leaf0 = query_ops[0].row
    lo0, _ = code.pair_indices(layer_pos[0], 0)
    ok = ok & jnp.all(comp_val == jnp.where(positions == lo0, leaf0[:, 0], leaf0[:, 1]))

    residual = jnp.full((block_len >> num_vars,), proof.final_poly)
    ok = ok & verify_fold_chain(code, query_ops, betas, layer_pos, residual)

    return t, ok


__all__ = ["verify_jagged_eval_msg", "stacked_basefold_verify"]
