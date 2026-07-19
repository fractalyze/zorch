# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`SumcheckKernel` native defaults: the per-round sumcheck ALGEBRA (message
components, state fold, verifier reduce/round-check) the core threads. These pin
the single-MLE degree-1 wire the choreography frames on top of."""

from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.poly.multilinear import eval_mle, mle_fold


def _ef(x: int) -> fnp.ndarray:
    return fnp.array(x, EF)


class ReduceClaimTest(absltest.TestCase):
    def test_additive_combine(self) -> None:
        # native: s(0) + r*s(1) -- NOT the affine (1-r)*s(0) + r*s(1) bind.
        k = SumcheckKernel()
        got = k.reduce_claim(_ef(0), (_ef(3), _ef(5)), _ef(2))
        self.assertEqual(got.tolist(), (_ef(3) + _ef(2) * _ef(5)).tolist())

    def test_ignores_running_claim(self) -> None:
        k = SumcheckKernel()
        a = k.reduce_claim(_ef(0), (_ef(3), _ef(5)), _ef(2))
        b = k.reduce_claim(_ef(99), (_ef(3), _ef(5)), _ef(2))
        self.assertEqual(a.tolist(), b.tolist())


class RoundCheckTest(absltest.TestCase):
    def test_consistent_claim_passes(self) -> None:
        k = SumcheckKernel()
        s0, s1, coord = _ef(3), _ef(5), _ef(7)
        one = fnp.array(1, EF)
        claim = (one - coord) * s0 + coord * s1
        self.assertTrue(bool(k.round_check(claim, (s0, s1), coord)))

    def test_tampered_claim_fails(self) -> None:
        k = SumcheckKernel()
        s0, s1, coord = _ef(3), _ef(5), _ef(7)
        one = fnp.array(1, EF)
        claim = (one - coord) * s0 + coord * s1 + one
        self.assertFalse(bool(k.round_check(claim, (s0, s1), coord)))


class MessageFoldTest(absltest.TestCase):
    def test_message_recovers_the_claim_at_the_bound_coord(self) -> None:
        # message must satisfy round_check against the claim it was built from:
        # eval_mle(mle, zs) == (1-zs[-1])*s0 + zs[-1]*s1.
        k = SumcheckKernel()
        mle = fnp.arange(1, 9, dtype=EF)  # 3 vars
        zs = fnp.array([2, 3, 4], EF)
        claim = eval_mle(mle, zs)
        s0, s1 = k.message((mle, claim, zs))
        self.assertTrue(bool(k.round_check(claim, (s0, s1), zs[-1])))

    def test_fold_advances_state(self) -> None:
        k = SumcheckKernel()
        mle = fnp.arange(1, 9, dtype=EF)
        zs = fnp.array([2, 3, 4], EF)
        claim = eval_mle(mle, zs)
        msg = k.message((mle, claim, zs))
        r = _ef(6)
        mle2, claim2, zs2 = k.fold((mle, claim, zs), msg, r)
        self.assertEqual(mle2.tolist(), mle_fold(mle, r).tolist())
        self.assertEqual(claim2.tolist(), (msg[0] + r * msg[1]).tolist())
        self.assertEqual(zs2.tolist(), zs[:-1].tolist())

    def test_final_default_none(self) -> None:
        self.assertIsNone(SumcheckKernel().final((fnp.arange(2, dtype=EF),)))

    def test_initial_state_carries_mle_claim_point(self) -> None:
        k = SumcheckKernel()
        mle = fnp.arange(1, 5, dtype=EF)
        zs = fnp.array([2, 3], EF)
        state = k.initial_state(mle, zs, _ef(9))
        self.assertEqual(state[0].tolist(), mle.tolist())
        self.assertEqual(state[1].tolist(), _ef(9).tolist())
        self.assertEqual(state[2].tolist(), zs.tolist())


if __name__ == "__main__":
    absltest.main()
