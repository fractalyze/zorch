# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.sumcheck.eq.eq_poly import prove_eq_poly
from zorch.sumcheck.eq.small_value import prove_eq_poly_small_value
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont

# Boolean weight vectors per variable count.
_W = {4: [1, 0, 1, 0], 6: [1, 0, 1, 1, 0, 1]}


class SmallValueTest(absltest.TestCase):
    def test_matches_eq_poly(self) -> None:
        # The small-value prover computes the same eq-weighted sumcheck as Algorithm
        # 5, so every round message must agree; identical messages keep the shared
        # transcript in lockstep, so same-seed runs compare directly.
        for d, l, l_0 in [(2, 4, 1), (2, 6, 2), (3, 6, 2), (2, 6, 1), (2, 6, 3)]:
            p = jnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
            w = jnp.array(_W[l], dtype=KB)
            _, _, ref = prove_eq_poly(p, w, cheap_transcript(KB))
            _, _, got = prove_eq_poly_small_value(p, w, l_0, cheap_transcript(KB))
            self.assertLen(got, l)
            for i, (a, b) in enumerate(zip(ref, got, strict=True)):
                self.assertTrue(
                    bool(jnp.array_equal(a, b)), msg=f"d={d} l={l} l_0={l_0} round {i}"
                )

    def test_prove_folds_to_scalar(self) -> None:
        p = jnp.arange(1, 2 * 64 + 1, dtype=KB).reshape(2, 64)
        p_final, _, msgs = prove_eq_poly_small_value(
            p, jnp.array(_W[6], dtype=KB), 2, cheap_transcript(KB)
        )
        self.assertEqual(p_final.shape, (2, 1))
        self.assertLen(msgs, 6)


if __name__ == "__main__":
    absltest.main()
