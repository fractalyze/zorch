# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.sumcheck.domain import product_round_poly
from zorch.sumcheck.eq.eq_poly import (
    EqPolyRound,
    _split_pairs,
    compute_eq_evaluations,
    prove_eq_poly,
)
from zorch.sumcheck.prover import SumcheckRound
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


class EqPolyTest(absltest.TestCase):
    def test_suffix_eq_tables(self) -> None:
        # w = [0, 1]: eq(w[-1:], ·) = [0, 1]; eq([0, 1], ·) = [0, 1, 0, 0].
        tables = compute_eq_evaluations(jnp.array([0, 1], dtype=KB))
        self.assertTrue(bool(jnp.array_equal(tables[0], jnp.array([0, 1], dtype=KB))))
        self.assertTrue(
            bool(jnp.array_equal(tables[1], jnp.array([0, 1, 0, 0], dtype=KB)))
        )

    def test_fold_matches_linear_update(self) -> None:
        # r=3 fold of P1 = 1..32: (P1[16+j] − P1[j])·3 + P1[j] = 49..64.
        P = jnp.stack([jnp.arange(1, 33, dtype=KB), jnp.arange(2, 34, dtype=KB)])
        rnd = EqPolyRound(SumcheckRound(degree=2), jnp.array([0, 1, 1, 0, 1], dtype=KB))
        state = (P, jnp.ones(1, dtype=KB))
        _, cache = rnd._round_poly(state)
        new_p, _ = rnd._fold(cache, state[1], jnp.array(3, dtype=KB))
        self.assertEqual(new_p.shape, (2, 16))
        self.assertTrue(bool(jnp.array_equal(new_p[0], jnp.arange(49, 65, dtype=KB))))

    def test_messages_match_product_sumcheck(self) -> None:
        # EqPolySC(P, w) round messages equal a product sumcheck over the factors
        # [P_1, …, P_d, eq(w, ·)], on Û_d — the top node u=d−1 the reference sends
        # is recovered by a verifier, so compare against ref[:-1].
        d, l = 3, 4
        P = jnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
        w = jnp.array([1, 0, 1, 0], dtype=KB)
        eq_w = compute_eq_evaluations(w)[-1]

        rnd = EqPolyRound(SumcheckRound(degree=d), w)
        state = (P, jnp.ones(1, dtype=KB))
        ref = jnp.concatenate([P, eq_w[None, :]], axis=0)

        for challenge in (2, 3, 5, 7):
            r = jnp.array(challenge, dtype=KB)
            msg, cache = rnd._round_poly(state)
            self.assertTrue(bool(jnp.array_equal(msg, product_round_poly(ref)[:-1])))
            state = rnd._fold(cache, state[1], r)
            ref_p0, ref_diff = _split_pairs(ref)
            ref = ref_diff * r + ref_p0

    def test_prove_folds_all_rounds(self) -> None:
        P = jnp.stack([jnp.arange(1, 33, dtype=KB), jnp.arange(2, 34, dtype=KB)])
        p_final, _, msgs = prove_eq_poly(
            P, jnp.array([0, 1, 1, 0, 1], dtype=KB), cheap_transcript(KB)
        )
        self.assertLen(msgs, 5)
        self.assertEqual(p_final.shape, (2, 1))
        for msg in msgs:
            self.assertEqual(msg.shape, (2,))

    def test_coeff_round_verifies(self) -> None:
        # The degree-(d+1) coefficient form of each round poly verifies against
        # CoeffsSumcheckRound; the verifier drives the challenges and the claim
        # reduces down to eq(w, r)·Π_k P_k(r) — a full sumcheck round trip.
        d, l = 2, 4
        P = jnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)
        w = jnp.array([1, 0, 1, 0], dtype=KB)
        claim = jnp.sum(
            expand_eq_to_hypercube(w, jnp.ones((), KB)) * jnp.prod(P, axis=0)
        )

        rnd = EqPolyRound(SumcheckRound(degree=d), w)
        state = (P, jnp.ones(1, dtype=KB))
        verifier = CoeffsSumcheckRound(degree=d + 1)
        transcript: Transcript = cheap_transcript(KB)
        point = []
        for _ in range(l):
            coeffs, cache = rnd._round_coeffs(state)
            self.assertEqual(coeffs.shape, (d + 2,))
            claim, transcript, r, ok = verifier(claim, coeffs, transcript)
            self.assertTrue(bool(ok))
            state = rnd._fold(cache, state[1], r)
            point.append(r)
        expected = eval_eq(w, jnp.stack(point)) * jnp.prod(state[0][:, 0])
        self.assertTrue(bool(claim == expected))


if __name__ == "__main__":
    absltest.main()
