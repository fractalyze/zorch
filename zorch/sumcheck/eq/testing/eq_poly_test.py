# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.challenge import ChallengePolicy
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.sumcheck.domain import product_round_poly
from zorch.sumcheck.eq.eq_poly import (
    EqPolyRound,
    _split_slope,
    compute_eq_evaluations,
    prove_eq_poly,
)
from zorch.sumcheck.prover import ProductSummand
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript
from zorch.verify import RunningClaim

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)


class EqPolyTest(absltest.TestCase):
    def test_suffix_eq_tables(self) -> None:
        # w = [0, 1]: eq(w[-1:], ·) = [0, 1]; eq([0, 1], ·) = [0, 1, 0, 0].
        tables = compute_eq_evaluations(fnp.array([0, 1], dtype=KB))
        self.assertTrue(bool(fnp.array_equal(tables[0], fnp.array([0, 1], dtype=KB))))
        self.assertTrue(
            bool(fnp.array_equal(tables[1], fnp.array([0, 1, 0, 0], dtype=KB)))
        )

    def test_fold_matches_linear_update(self) -> None:
        # r=3 fold of P1 = 1..32: (P1[16+j] − P1[j])·3 + P1[j] = 49..64.
        P = fnp.stack([fnp.arange(1, 33, dtype=KB), fnp.arange(2, 34, dtype=KB)])
        rnd = EqPolyRound(
            ProductSummand(degree=2),
            fnp.array([0, 1, 1, 0, 1], dtype=KB),
            challenges=_CH,
        )
        state = (P, fnp.ones(1, dtype=KB))
        _, cache = rnd._round_poly(state)
        new_p, _ = rnd._fold(cache, state[1], fnp.array(3, dtype=KB))
        self.assertEqual(new_p.shape, (2, 16))
        self.assertTrue(bool(fnp.array_equal(new_p[0], fnp.arange(49, 65, dtype=KB))))

    def test_messages_match_product_sumcheck(self) -> None:
        # EqPolySC(P, w) round messages equal a product sumcheck over the factors
        # [P_1, …, P_d, eq(w, ·)], on Û_d — the top node u=d−1 the reference sends
        # is recovered by a verifier, so compare against ref[:-1].
        d, l = 3, 4
        P = fnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
        w = fnp.array([1, 0, 1, 0], dtype=KB)
        eq_w = compute_eq_evaluations(w)[-1]

        rnd = EqPolyRound(ProductSummand(degree=d), w, challenges=_CH)
        state = (P, fnp.ones(1, dtype=KB))
        ref = fnp.concatenate([P, eq_w[None, :]], axis=0)

        for challenge in (2, 3, 5, 7):
            r = fnp.array(challenge, dtype=KB)
            msg, cache = rnd._round_poly(state)
            self.assertTrue(bool(fnp.array_equal(msg, product_round_poly(ref)[:-1])))
            state = rnd._fold(cache, state[1], r)
            ref_p0, ref_diff = _split_slope(ref)
            ref = ref_diff * r + ref_p0

    def test_prove_folds_all_rounds(self) -> None:
        P = fnp.stack([fnp.arange(1, 33, dtype=KB), fnp.arange(2, 34, dtype=KB)])
        w = fnp.array([0, 1, 1, 0, 1], dtype=KB)
        claim = fnp.sum(
            expand_eq_to_hypercube(w, fnp.ones((), KB)) * fnp.prod(P, axis=0)
        )
        carry, _, msgs = prove_eq_poly(
            P, w, claim, cheap_transcript(KB), challenges=_CH
        )
        self.assertLen(msgs, 5)
        self.assertEqual(carry.state[0].shape, (2, 1))
        for msg in msgs:
            self.assertEqual(msg.shape, (2,))

    def test_coeff_round_verifies(self) -> None:
        # The degree-(d+1) coefficient form of each round poly verifies against
        # CoeffsSumcheckRound; the verifier drives the challenges and the claim
        # reduces down to eq(w, r)·Π_k P_k(r) — a full sumcheck round trip.
        d, l = 2, 4
        P = fnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
        w = fnp.array([1, 0, 1, 0], dtype=KB)
        claim = fnp.sum(
            expand_eq_to_hypercube(w, fnp.ones((), KB)) * fnp.prod(P, axis=0)
        )

        rnd = EqPolyRound(ProductSummand(degree=d), w, challenges=_CH)
        state = (P, fnp.ones(1, dtype=KB))
        verifier = CoeffsSumcheckRound(degree=d + 1, challenges=_CH)
        transcript: Transcript = cheap_transcript(KB)
        running = RunningClaim(claim, fnp.zeros((l,), claim.dtype), fnp.int32(0))
        for i in range(l):
            coeffs, cache = rnd._round_coeffs(state)
            self.assertEqual(coeffs.shape, (d + 2,))
            running, transcript, ok = verifier(running, transcript, coeffs)
            self.assertTrue(bool(ok))
            state = rnd._fold(cache, state[1], running.point[i])
        claim = running.value
        expected = eval_eq(w, running.point) * fnp.prod(state[0][:, 0])
        self.assertTrue(bool(claim == expected))


if __name__ == "__main__":
    absltest.main()
