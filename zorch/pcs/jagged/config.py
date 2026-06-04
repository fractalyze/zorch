# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged opening proof — the wire payload `JaggedPcsProver.open` emits and
`JaggedPcsVerifier.verify` consumes.

The opening composes four sub-proofs in one Fiat-Shamir transcript: the outer
Hadamard sumcheck `Σ_i D(i)·J̃(i) = claim` (`outer_sumcheck_polys`), the inner
jagged-assist sumcheck reproving `J̃(z_row, z_col, z_final)`
(`inner_round_polys`), and the stacked BaseFold opening of the dense MLE at
`z_final` (`dense_eval`/`column_values`/`basefold_proof`). The reduced scalar
`outer_final_eval` (= D(z_final)·J̃(z_final), the outer sumcheck's final claim)
rides along as the product-check input; `inner_claimed_sum`
(= J̃(z_row, z_col, z_final)) is the seed the verifier feeds to the stock
`verify(SumcheckRound(2))` over `inner_round_polys` and the S5 product check —
a forged value fails the inner leaf check / S5 chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from jax import Array
from jax.tree_util import register_dataclass

from zorch.pcs.basefold.config import BasefoldProof


@partial(
    register_dataclass,
    data_fields=[
        "outer_sumcheck_polys",
        "outer_final_eval",
        "inner_round_polys",
        "inner_claimed_sum",
        "dense_eval",
        "column_values",
        "basefold_root",
        "basefold_proof",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class JaggedOpeningProof:
    """One jagged PCS opening.

    outer_sumcheck_polys: `[n_d, degree+1]` round polynomials of the outer
        Hadamard sumcheck (stock evaluation form; degree 2 → 3 evals/round).
    outer_final_eval: the outer sumcheck's reduced final claim
        `D(z_final)·J̃(z_final)`, the product the verifier checks.
    inner_round_polys: `[2·n_d, 3]` eval-form round polynomials of the inner
        jagged-assist sumcheck, replayed by the stock `verify(SumcheckRound(2))`.
    inner_claimed_sum: `J̃(z_row, z_col, z_final)`, the seed the verifier feeds to
        the stock inner `verify` and the S5 product check (a forged value fails
        the inner leaf check / S5 chain).
    dense_eval: `D(z_final)` the stacked BaseFold opening proves.
    column_values: `(K,)` per-column BaseFold evals at `stack_point`.
    basefold_root: the un-bound BaseFold matrix-commitment root. The jagged
        commitment binds it as `compress([root, hash(structure)])`; the verifier
        re-derives the bind from this root (S0) and feeds it to the stacked
        BaseFold verify (S6, which binds it via Fiat-Shamir).
    basefold_proof: the underlying BaseFold opening proof.

    A registered pytree so it crosses the open/verify `@jit` boundary as one
    leaf-bearing container.
    """

    outer_sumcheck_polys: Array
    outer_final_eval: Array
    inner_round_polys: Array
    inner_claimed_sum: Array
    dense_eval: Array
    column_values: Array
    basefold_root: Array
    basefold_proof: BasefoldProof
