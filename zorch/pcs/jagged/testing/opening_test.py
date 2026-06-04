# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged PCS opening round-trip + tamper negatives.

Commit small variable-height blocks, compute the true per-column claims
`{MLE_c(z_row)}` from the block data so the opening identity holds, then
`open` -> `verify == True`. Anchors the prover-side values (dense_eval ==
eval_mle(D, z_final); inner_claimed_sum == eval_jagged_mle(...)) and exercises
four tamper paths (each must reject).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.prover import BasefoldProver
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.pcs.jagged.config import JaggedOpeningProof
from zorch.pcs.jagged.poly import eval_jagged_mle
from zorch.pcs.jagged.prover import (
    JaggedPcsProver,
    JaggedProverData,
    _compress_column_claims,
)
from zorch.pcs.jagged.verifier import JaggedPcsVerifier
from zorch.poly.multilinear import eval_mle
from zorch.sumcheck.verifier import SumcheckRound as VSumcheckRound
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript
from zorch.verify import verify as outer_verify

# heights [4, 2, 3], width-1 columns -> area 9, n_d 5; log_stacking_height 5
# makes log_m == n_d (the seam's requirement), K == 1.
_HEIGHTS = [4, 2, 3]
_LOG_STACK = 5
_N_R = 2  # log2_ceil(max height 4)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return rand_field(seed, (*shape, 4), F).view(EF).reshape(shape)


def _t() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _stack() -> tuple[JaggedPcsProver, JaggedPcsVerifier]:
    log_s = _LOG_STACK
    S = 1 << log_s
    rs = ReedSolomon(message_len=S, blowup=2, dtype=EF)
    sponge, comp, tree = koalabear16_merkle()
    prover = JaggedPcsProver(BasefoldProver(rs, tree, num_queries=4), sponge, comp)
    verifier = JaggedPcsVerifier(
        BasefoldVerifier(rs, tree, num_queries=4), sponge, comp
    )
    return prover, verifier


def _blocks(seed: int) -> list[jnp.ndarray]:
    """One width-1 block per height, with random EF entries."""
    return [_rand_ef(seed + i, (h, 1)) for i, h in enumerate(_HEIGHTS)]


def _column_claims(blocks: list[jnp.ndarray], z_row: jnp.ndarray) -> jnp.ndarray:
    """True {MLE_c(z_row)} per unit-width column, padded to 2^n_r rows."""
    row_len = 1 << _N_R
    claims = []
    for b in blocks:
        for w in range(b.shape[1]):
            col = b[:, w]
            pad = jnp.zeros(row_len - col.shape[0], dtype=col.dtype)
            col_padded = jnp.concatenate([col, pad])
            claims.append(eval_mle(col_padded, z_row))
    return jnp.stack(claims)


class JaggedOpeningTest(absltest.TestCase):
    def _open(self, seed: int) -> tuple[
        JaggedPcsVerifier,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        JaggedProverData,
        JaggedOpeningProof,
    ]:
        prover, verifier = _stack()
        blocks = _blocks(seed)
        commitment, pdata = prover.commit(blocks, log_stacking_height=_LOG_STACK)
        z_row = _rand_ef(seed + 50, (_N_R,))
        column_claims = _column_claims(blocks, z_row)
        proof, _ = prover.open(pdata, z_row, column_claims, _t())
        return verifier, commitment, z_row, column_claims, pdata, proof

    def test_round_trip_verifies(self) -> None:
        verifier, commitment, z_row, column_claims, pdata, proof = self._open(1)
        ok, _ = verifier.verify(
            commitment, z_row, column_claims, pdata.layout, proof, _t()
        )
        self.assertTrue(bool(ok))

    def test_prover_anchors(self) -> None:
        # dense_eval == eval_mle(D, z_final); inner_claimed_sum ==
        # eval_jagged_mle(...). Reconstruct z_final from the outer round polys.
        verifier, _commitment, z_row, _claims, pdata, proof = self._open(2)
        cfg = pdata.cfg
        z_col_seed_t = _t()
        z_col_seed_t, z_col = z_col_seed_t.sample(cfg.n_c)
        z_col = z_col.astype(cfg.dtype)  # mirror the prover's EF embedding
        # Replay the outer sumcheck to recover z_final (verifier's path).
        claim = _compress_column_claims(_claims, z_col)
        point, _final, _tr, _ok = outer_verify(
            VSumcheckRound(2), claim, proof.outer_sumcheck_polys, z_col_seed_t
        )
        z_final = point.astype(cfg.dtype)
        # D(z_final) anchor.
        self.assertEqual(
            proof.dense_eval.tolist(), eval_mle(pdata.dense, z_final).tolist()
        )
        # J̃(z_row, z_col, z_final) anchor.
        want_inner = eval_jagged_mle(
            pdata.col_prefix_sums, z_row, z_col, z_final, cfg=cfg
        )
        self.assertEqual(proof.inner_claimed_sum.tolist(), want_inner.tolist())

    def test_tampered_outer_round_poly_rejects(self) -> None:
        verifier, commitment, z_row, column_claims, pdata, proof = self._open(3)
        bad_polys = proof.outer_sumcheck_polys.at[0, 0].add(jnp.array(1, EF))
        bad = dataclasses.replace(proof, outer_sumcheck_polys=bad_polys)
        ok, _ = verifier.verify(
            commitment, z_row, column_claims, pdata.layout, bad, _t()
        )
        self.assertFalse(bool(ok))

    def test_tampered_dense_eval_rejects(self) -> None:
        verifier, commitment, z_row, column_claims, pdata, proof = self._open(4)
        bad = dataclasses.replace(proof, dense_eval=proof.dense_eval + jnp.array(1, EF))
        ok, _ = verifier.verify(
            commitment, z_row, column_claims, pdata.layout, bad, _t()
        )
        self.assertFalse(bool(ok))

    def test_tampered_column_value_rejects(self) -> None:
        verifier, commitment, z_row, column_claims, pdata, proof = self._open(5)
        bad_vals = proof.column_values.at[0].add(jnp.array(1, EF))
        bad = dataclasses.replace(proof, column_values=bad_vals)
        ok, _ = verifier.verify(
            commitment, z_row, column_claims, pdata.layout, bad, _t()
        )
        self.assertFalse(bool(ok))

    def test_tampered_inner_proof_rejects(self) -> None:
        verifier, commitment, z_row, column_claims, pdata, proof = self._open(6)
        inner = proof.inner_proof
        bad_inner = dataclasses.replace(
            inner, round_polys=inner.round_polys.at[0, 0].add(jnp.array(1, EF))
        )
        bad = dataclasses.replace(proof, inner_proof=bad_inner)
        ok, _ = verifier.verify(
            commitment, z_row, column_claims, pdata.layout, bad, _t()
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
