# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.prover import LogupSummand, logup_combine
from zorch.prove import fold_rounds
from zorch.round import RunningClaim
from zorch.sumcheck.domain import natural_domain, subgroup_sum
from zorch.sumcheck.prover import ProductSummand, StandardRound, initial_claim
from zorch.sumcheck.sqrt_space import prove_sqrt_space
from zorch.sumcheck.stage import SumcheckWitness, SumClaim
from zorch.sumcheck.univariate_skip import (
    UnivariateSkipProver,
    UnivariateSkipVerifier,
    prove_univariate_skip,
    round0_message,
    skip_round0,
    verify_univariate_skip,
)
from zorch.testkit.fusion import _FUSION_SAFE
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)
KBx4 = zk_dtypes.koalabearx4_mont


def _claim(p: fnp.ndarray) -> fnp.ndarray:
    return fnp.sum(fnp.prod(p, axis=0))


class Round0Test(parameterized.TestCase):
    @parameterized.parameters((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3))
    def test_subgroup_sum_equals_boolean_claim(self, skip: int, m: int) -> None:
        # The skip's soundness rests on Σ_{z∈D} s₀(z) == the original boolean-hypercube
        # sum: identifying the 2^skip boolean prefixes with the |D| subgroup points, the
        # subgroup sum of the round-0 message must reproduce the claim. This is the
        # F₁₇ worked example (wiki `[[univariate-skip]]`) over the real field.
        p = rand_field(100 + 10 * skip + m, (m, 1 << 4), KB)
        msg0, _ = round0_message(p, skip, ProductSummand(degree=m))
        self.assertTrue(bool(subgroup_sum(msg0, skip) == _claim(p)))

    def test_round0_message_is_base_field(self) -> None:
        # Round 0 runs entirely in the base field — extension arithmetic starts only
        # once r₀ is bound at round 1.
        p = rand_field(5, (2, 1 << 4), KB)
        msg0, coeffs_z = round0_message(p, 2, ProductSummand(degree=2))
        self.assertEqual(msg0.dtype, KB)
        self.assertEqual(coeffs_z.dtype, KB)


class SkipRoundTripTest(parameterized.TestCase):
    def test_stage_round_trip_and_transcript_agreement(self) -> None:
        total = 5
        state = rand_field(200, (2, 1 << total), KB)
        claim = _claim(state)
        policy = ChallengePolicy(KBx4)
        prover = UnivariateSkipProver(2, ProductSummand(2), challenges=policy)
        verifier = UnivariateSkipVerifier(2, ProductSummand(2), challenges=policy)
        source_claim = SumClaim(claim, total)
        proved = prover.prove(
            source_claim, SumcheckWitness(state), cheap_transcript(KB)
        )
        verified = verifier.verify(
            source_claim,
            proved.reduction_proof,
            cheap_transcript(KB),
        )
        self.assertTrue(bool(verified.ok))
        self.assertEqual(proved.reduced_claim.prism_point.shape, (1 + total - 2,))
        self.assertTrue(
            bool(
                fnp.all(
                    proved.reduced_claim.prism_point
                    == verified.reduced_claim.prism_point
                )
            )
        )
        self.assertTrue(
            bool(proved.reduced_claim.value == verified.reduced_claim.value)
        )
        _, prover_next = proved.transcript.sample(1)
        _, verifier_next = verified.transcript.sample(1)
        self.assertTrue(bool(fnp.all(prover_next == verifier_next)))

    @parameterized.parameters((1, 1), (2, 1), (1, 2), (2, 2))
    def test_base_field_round_trip(self, skip: int, m: int) -> None:
        total = 5
        p = rand_field(7 + skip + m, (m, 1 << total), KB)
        summand = ProductSummand(degree=m)
        carry, _, msgs = prove_univariate_skip(
            p, _claim(p), skip, cheap_transcript(KB), summand, challenges=_CH
        )
        reduced, _, point, ok = verify_univariate_skip(
            _claim(p), msgs, skip, total, cheap_transcript(KB), degree=m, challenges=_CH
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(reduced == fnp.prod(carry.state[:, 0])))
        # Round + challenge count both drop to 1 + n.
        self.assertLen(msgs, 1 + (total - skip))
        self.assertLen(point, 1 + (total - skip))

    @parameterized.parameters((1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (4, 2))
    def test_extension_field_round_trip(self, skip: int, m: int) -> None:
        total = 6
        p = rand_field(50 + skip + m, (m, 1 << total), KB)
        summand = ProductSummand(degree=m)
        carry, _, msgs = prove_univariate_skip(
            p,
            _claim(p),
            skip,
            cheap_transcript(KB),
            summand,
            challenges=ChallengePolicy(KBx4),
        )
        reduced, _, _, ok = verify_univariate_skip(
            _claim(p),
            msgs,
            skip,
            total,
            cheap_transcript(KB),
            degree=m,
            challenges=ChallengePolicy(KBx4),
        )
        # Extension arithmetic entered at round 1: the folded state is extension-field.
        self.assertEqual(carry.state.dtype, KBx4)
        self.assertTrue(bool(ok))
        self.assertTrue(bool(reduced == fnp.prod(carry.state[:, 0])))


class SkipZeroTest(absltest.TestCase):
    def test_skip0_byte_identical_to_standard_run(self) -> None:
        # skip=0 is the off switch: byte-identical proof + transcript to the plain
        # StandardRound run, a strict opt-in extension rather than a fork.
        m, total = 2, 4
        p = rand_field(3, (m, 1 << total), KB)
        summand = ProductSummand(degree=m)
        carry0, _, msgs0 = prove_univariate_skip(
            p, _claim(p), 0, cheap_transcript(KB), summand, challenges=_CH
        )
        f_ref, _, msgs_ref = fold_rounds(
            StandardRound(summand, challenges=_CH),
            initial_claim(p, _claim(p), total),
            cheap_transcript(KB),
            total,
        )
        self.assertLen(msgs0, total)
        for a, b in zip(msgs0, msgs_ref, strict=True):
            self.assertTrue(bool(fnp.array_equal(a, b)))
        self.assertTrue(bool(fnp.array_equal(carry0.state, f_ref.state)))

    def test_skip0_verifies(self) -> None:
        m, total = 2, 4
        p = rand_field(4, (m, 1 << total), KB)
        carry0, _, msgs0 = prove_univariate_skip(
            p,
            _claim(p),
            0,
            cheap_transcript(KB),
            ProductSummand(degree=m),
            challenges=_CH,
        )
        reduced, _, point, ok = verify_univariate_skip(
            _claim(p), msgs0, 0, total, cheap_transcript(KB), degree=m, challenges=_CH
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(reduced == fnp.prod(carry0.state[:, 0])))
        self.assertLen(point, total)


class SkipSoundnessTest(absltest.TestCase):
    def _proof(self) -> tuple[fnp.ndarray, int, int, list[fnp.ndarray]]:
        m, total = 2, 4
        p = rand_field(11, (m, 1 << total), KB)
        _, _, msgs = prove_univariate_skip(
            p,
            _claim(p),
            2,
            cheap_transcript(KB),
            ProductSummand(degree=m),
            challenges=ChallengePolicy(KBx4),
        )
        return p, total, m, msgs

    def test_tampered_round0_message_rejected(self) -> None:
        p, total, m, msgs = self._proof()
        bad = list(msgs)
        bad[0] = bad[0].at[0].add(fnp.ones((), KB))
        _, _, _, ok = verify_univariate_skip(
            _claim(p),
            bad,
            2,
            total,
            cheap_transcript(KB),
            degree=m,
            challenges=ChallengePolicy(KBx4),
        )
        self.assertFalse(bool(ok))

    def test_wrong_claim_rejected(self) -> None:
        p, total, m, msgs = self._proof()
        _, _, _, ok = verify_univariate_skip(
            _claim(p) + fnp.ones((), KB),
            msgs,
            2,
            total,
            cheap_transcript(KB),
            degree=m,
            challenges=ChallengePolicy(KBx4),
        )
        self.assertFalse(bool(ok))


class ComposedTailTest(parameterized.TestCase):
    @parameterized.parameters((1, 2), (2, 2), (2, 1))
    def test_skip_then_sqrt_space_tail_stacks(self, skip: int, m: int) -> None:
        # Run the tail under the √-space engine instead of StandardRound. √-space
        # reproduces the standard round messages, so the composed proof must match the
        # plain-tail reference message-for-message and still verify.
        total = 6
        p = rand_field(70 + skip + m, (m, 1 << total), KB)
        summand = ProductSummand(degree=m)
        c = _claim(p)

        # Reference: skip round 0 + plain StandardRound tail.
        _, _, ref_msgs = prove_univariate_skip(
            p,
            c,
            skip,
            cheap_transcript(KB),
            summand,
            challenges=ChallengePolicy(KBx4),
        )

        # Composed: same round 0, then the √-space tail over the bound extension state.
        start = RunningClaim(c, fnp.zeros((total - skip + 1,), KBx4), fnp.int32(0))
        carry, transcript, msg0 = skip_round0(
            p, start, skip, cheap_transcript(KB), summand, ChallengePolicy(KBx4)
        )
        final, transcript, tail = prove_sqrt_space(
            carry.state,
            carry.claim.value,
            transcript,
            summand,
            domain=natural_domain(m, KBx4),
            challenges=ChallengePolicy(KBx4),
        )
        composed = [msg0] + tail

        self.assertLen(composed, len(ref_msgs))
        for a, b in zip(composed, ref_msgs, strict=True):
            self.assertTrue(bool(fnp.array_equal(a, b)))
        reduced, _, _, ok = verify_univariate_skip(
            c,
            composed,
            skip,
            total,
            cheap_transcript(KB),
            degree=m,
            challenges=ChallengePolicy(KBx4),
        )
        self.assertTrue(bool(ok))
        # The √-space tail's own folded state, not round 0's.
        self.assertTrue(bool(reduced == fnp.prod(final.state[:, 0])))


class SummandGenericTest(parameterized.TestCase):
    @parameterized.parameters((1,), (2,), (4,))
    def test_logup_summand_drops_into_skip(self, skip_rounds: int) -> None:
        # A non-product summand (LogUp: `eq*(λ*(n0*d1+n1*d0)+d0*d1)`, degree 3 over 5
        # factors) drives the skip unchanged — round 0 combines via combine_scalars, the
        # tail is StandardRound(ext) — exercising the seam beyond the product summand.
        total = 6
        p = rand_field(400 + skip_rounds, (LogupSummand.NUM_FACTORS, 1 << total), KB)
        lam = rand_field(7, (), KB)
        summand = LogupSummand(lam)
        c = fnp.sum(logup_combine(lam, *p))
        pf, _, msgs = prove_univariate_skip(
            p,
            c,
            skip_rounds,
            cheap_transcript(KB),
            summand,
            challenges=ChallengePolicy(KBx4),
        )
        reduced, _, point, ok = verify_univariate_skip(
            c,
            msgs,
            skip_rounds,
            total,
            cheap_transcript(KB),
            degree=3,
            challenges=ChallengePolicy(KBx4),
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(reduced == logup_combine(lam, *pf.state[:, 0])))
        self.assertLen(msgs, 1 + total - skip_rounds)


class Round0FusionTest(absltest.TestCase):
    def test_round0_lowers_to_bounded_kernel_set(self) -> None:
        # Round 0 cannot be one straight-line kernel (it is inner-sum + iNTT), but it
        # must lower to a BOUNDED fused-kernel set: NTT kernels + element-wise combine +
        # the one inherent Σ over H_n — no gather/scatter/dot/while boundary. Whether
        # the NTTs fuse into a single XLA kernel is the open
        # XLA-side question (wiki `[[univariate-skip]]`); the
        # authoritative gate is the XLA rewriter, this is a
        # cheap HLO-shape proxy mirroring `testkit.fusion.assert_fusion_ready`.
        p = rand_field(6, (2, 1 << 5), KB)
        hlo = (
            frx.jit(lambda x: round0_message(x, 2, ProductSummand(degree=2)))
            .lower(p)
            .as_text()
        )
        ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
        allowed = _FUSION_SAFE | {"ntt", "reduce"}
        offenders = sorted({o for o in ops if o not in allowed})
        self.assertEqual(offenders, [], f"boundary ops in round 0: {offenders}")
        # Three transforms: factor iNTT, LDE NTT, s₀ iNTT — a bounded set.
        self.assertEqual(ops.count("ntt"), 3)


if __name__ == "__main__":
    absltest.main()
