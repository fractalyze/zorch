# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.jagged.poly import (
    JaggedStaticConfig,
    build_jagged_layout,
    eval_jagged_mle,
)
from zorch.pcs.jagged.prover import prove_jagged_eval
from zorch.pcs.jagged.verifier import verify_jagged_eval
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript

EF = zk_dtypes.koalabearx4_mont


def _t() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _field_val(x: Array) -> bytes:
    """Field scalar → bytes; int() is unavailable for koalabearx4_mont."""
    return np.array(x).tobytes()


def _layout() -> tuple[Array, JaggedStaticConfig]:
    # heights [4, 2, 3] → prefix [0, 4, 6, 9], total_area 9, n_d 5; l_max 4, n_r 3.
    return build_jagged_layout([4, 2, 3], l_max=4, n_r=3, dtype=EF)


class InnerSumcheckTest(absltest.TestCase):
    def test_inner_sumcheck_matches_eval_jagged_mle(self) -> None:
        cps, cfg = _layout()
        z_row = rand_field(1, (cfg.n_r,), EF)
        z_col = rand_field(2, (cfg.n_c,), EF)
        z_final = rand_field(3, (cfg.n_d,), EF)

        round_polys, claimed_sum, _ = prove_jagged_eval(
            cps, z_row, z_col, z_final, cfg=cfg, transcript=_t()
        )
        want = eval_jagged_mle(cps, z_row, z_col, z_final, cfg=cfg)
        # Tautological: prove_jagged_eval sets claimed_sum = eval_jagged_mle, so
        # this can never fail. The oracle anchor is the verify == True check below,
        # which the verifier passes only when the seed it is given (claimed_sum)
        # is the true jagged eval (the inner leaf check ties it down).
        self.assertEqual(_field_val(claimed_sum), _field_val(want))

        ok, _ = verify_jagged_eval(
            cps,
            z_row,
            z_col,
            z_final,
            round_polys,
            claimed_sum,
            cfg=cfg,
            transcript=_t(),
        )
        self.assertTrue(bool(ok))

    def test_tampered_round_poly_rejects(self) -> None:
        cps, cfg = _layout()
        z_row = rand_field(11, (cfg.n_r,), EF)
        z_col = rand_field(12, (cfg.n_c,), EF)
        z_final = rand_field(13, (cfg.n_d,), EF)

        round_polys, claimed_sum, _ = prove_jagged_eval(
            cps, z_row, z_col, z_final, cfg=cfg, transcript=_t()
        )
        # Perturb the first round poly's s(0) eval by +1.
        tampered = round_polys.at[0, 0].add(np.array(1, dtype=EF))

        ok, _ = verify_jagged_eval(
            cps, z_row, z_col, z_final, tampered, claimed_sum, cfg=cfg, transcript=_t()
        )
        self.assertFalse(bool(ok))

    def test_tampered_last_round_poly_rejects(self) -> None:
        # Perturb the last round poly's s(2) eval by +1.  This shifts s(0)+s(1) for
        # the last round, so the per-round consistency check for round[-1] fires
        # before the leaf check.  Rejection here ensures a bug in the final-round
        # folding (wrong claim delivered to the leaf check) would also be caught.
        cps, cfg = _layout()
        z_row = rand_field(21, (cfg.n_r,), EF)
        z_col = rand_field(22, (cfg.n_c,), EF)
        z_final = rand_field(23, (cfg.n_d,), EF)

        round_polys, claimed_sum, _ = prove_jagged_eval(
            cps, z_row, z_col, z_final, cfg=cfg, transcript=_t()
        )
        tampered = round_polys.at[-1, 2].add(np.array(1, dtype=EF))

        ok, _ = verify_jagged_eval(
            cps, z_row, z_col, z_final, tampered, claimed_sum, cfg=cfg, transcript=_t()
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
