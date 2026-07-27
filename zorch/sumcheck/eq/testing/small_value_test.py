# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.eq.eq_poly import prove_eq_poly
from zorch.sumcheck.eq.small_value import prove_eq_poly_small_value
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)

# Boolean weight vectors per variable count.
_W = {4: [1, 0, 1, 0], 6: [1, 0, 1, 1, 0, 1]}


def _eq_claim(factors: Array, w: Array) -> Array:
    """The eq-weighted sum the engine reduces."""
    ones = fnp.ones((), factors.dtype)
    return fnp.sum(fnp.prod(factors, axis=0) * expand_eq_to_hypercube(w, ones))


class SmallValueTest(absltest.TestCase):
    def test_matches_eq_poly(self) -> None:
        # The small-value prover computes the same eq-weighted sumcheck as Algorithm
        # 5, so every round message must agree; identical messages keep the shared
        # transcript in lockstep, so same-seed runs compare directly.
        for d, l, l_0 in [(2, 4, 1), (2, 6, 2), (3, 6, 2), (2, 6, 1), (2, 6, 3)]:
            p = fnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
            w = fnp.array(_W[l], dtype=KB)
            _, _, ref = prove_eq_poly(
                p, w, _eq_claim(p, w), cheap_transcript(KB), challenges=_CH
            )
            _, _, got = prove_eq_poly_small_value(
                p, w, l_0, _eq_claim(p, w), cheap_transcript(KB), challenges=_CH
            )
            self.assertLen(got, l)
            for i, (a, b) in enumerate(zip(ref, got, strict=True)):
                self.assertTrue(
                    bool(fnp.array_equal(a, b)), msg=f"d={d} l={l} l_0={l_0} round {i}"
                )

    def test_prove_folds_to_scalar(self) -> None:
        p = fnp.arange(1, 2 * 64 + 1, dtype=KB).reshape(2, 64)
        w = fnp.array(_W[6], dtype=KB)
        carry, _, msgs = prove_eq_poly_small_value(
            p, w, 2, _eq_claim(p, w), cheap_transcript(KB), challenges=_CH
        )
        self.assertEqual(carry.state[0].shape, (2, 1))
        self.assertLen(msgs, 6)


if __name__ == "__main__":
    absltest.main()
