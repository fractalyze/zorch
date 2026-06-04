# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged PCS verifier — the dual of `JaggedPcsProver.open`.

`JaggedPcsVerifier.verify` replays the four-step opening in the SAME Fiat-Shamir
order the prover wrote it, so the transcripts cannot diverge:

* S1: sample `z_col`, recompute `claim = Σ_c eq(z_col, c)·column_claims[c]`.
* S3: replay the outer Hadamard sumcheck against `claim`, reducing to `z_final`
  and the final claim `D(z_final)·J̃(z_final)`.
* S4: replay the inner jagged-assist sumcheck, re-deriving `J̃(z_row, z_col,
  z_final)`.
* S5: product check `dense_eval · jagged_eval == outer_final_eval`.
* S6: stacked BaseFold verify of `D(z_final)` against the un-bound root.
* S0: re-derive the structure bind `compress([root, hash(structure)])` and check
  it matches the commitment — the soundness anchor that the opened dense MLE is
  the one whose jagged structure was committed.

All flags are ANDed; one false anywhere rejects.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.pcs.jagged.config import JaggedOpeningProof
from zorch.pcs.jagged.dense import JaggedLayout
from zorch.pcs.jagged.inner_sumcheck import verify_jagged_eval
from zorch.pcs.jagged.prover import (
    _compress_column_claims,
    _indicator_inputs,
    _structure_vec,
)
from zorch.pcs.jagged.stacked import stacked_verify
from zorch.sumcheck.verifier import SumcheckRound
from zorch.transcript import Transcript
from zorch.verify import verify

# Outer Hadamard sumcheck degree (must match the prover's).
_OUTER_DEGREE = 2


class JaggedPcsVerifier:
    """Jagged PCS verifier. Holds the BaseFold verifier (O(1) verifier key) and
    the `sponge` / `compressor` that re-derive the structure bind (the jagged
    layer owns the bind, mirroring `JaggedPcsProver`)."""

    def __init__(
        self, bf_verifier: BasefoldVerifier, sponge: Sponge, compressor: Compression
    ) -> None:
        self.bf_verifier = bf_verifier
        self.sponge = sponge
        self.compressor = compressor

    def verify(
        self,
        commitment: Array,
        z_row: Array,
        column_claims: Array,
        structure: JaggedLayout,
        proof: JaggedOpeningProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Verify a jagged opening at `z_row`. `structure` is the committed
        region's `JaggedLayout`; `column_claims` is the `(L,)` per-column claim
        vector. Returns `(ok, transcript)`."""
        col_prefix_sums, _offsets, cfg = _indicator_inputs(structure)
        if cfg.n_d != structure.log_m:
            raise ValueError(
                f"jagged verify requires cfg.n_d == layout.log_m; got "
                f"n_d={cfg.n_d}, log_m={structure.log_m}."
            )

        # S1: sample z_col, recompute the column claim. Embed the base-field
        # squeeze in the indicator's extension field (mirrors the prover).
        transcript, z_col = transcript.sample(cfg.n_c)
        z_col = z_col.astype(cfg.dtype)
        claim = _compress_column_claims(column_claims, z_col)

        # S3: replay the outer Hadamard sumcheck.
        point, outer_final_eval, transcript, ok_outer = verify(
            SumcheckRound(_OUTER_DEGREE),
            claim,
            proof.outer_sumcheck_polys,
            transcript,
        )
        # No reversal: zorch's stock sumcheck folds MSB-first (see the prover).
        z_final = point.astype(cfg.dtype)

        # S4: replay the inner jagged-assist sumcheck.
        jagged_eval, ok_inner, transcript = verify_jagged_eval(
            col_prefix_sums,
            z_row,
            z_col,
            z_final,
            proof.inner_proof,
            cfg=cfg,
            transcript=transcript,
        )

        # S5: product check. The replayed final claim and the proof's stored
        # value must both equal D(z_final)·J̃(z_final).
        ok_prod = (proof.dense_eval * jagged_eval == outer_final_eval) & (
            proof.outer_final_eval == outer_final_eval
        )

        # S6: stacked BaseFold verify of D(z_final) against the un-bound root.
        ok_bf, transcript = stacked_verify(
            self.bf_verifier,
            proof.basefold_root,
            z_final,
            proof.dense_eval,
            proof.column_values,
            proof.basefold_proof,
            structure,
            transcript,
        )

        # S0: re-derive the structure bind and check it matches the commitment.
        structure_hash = self.sponge.hash(_structure_vec(structure))
        bound = self.compressor.compress(
            jnp.stack([proof.basefold_root, structure_hash])
        )
        ok_bind = jnp.all(bound == commitment)

        ok = ok_outer & ok_inner & ok_prod & ok_bf & ok_bind
        return ok, transcript
