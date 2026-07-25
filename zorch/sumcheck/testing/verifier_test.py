# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.prove import fold_rounds
from zorch.sumcheck import prover, verifier
from zorch.sumcheck.testing import eval_mle_oracle, product
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.verify import RunningClaim, verify

KB = zk_dtypes.koalabear_mont

# The transcript's own field: the schedule these tests pinned before the
# policy required an explicit field.
_CH = ChallengePolicy(KB)
_GPU_BACKEND = frx.default_backend() == "gpu"


def _standard(degree: int, challenges: ChallengePolicy = _CH) -> prover.StandardRound:
    """The product `StandardRound` at `degree`, the value-form dual of
    `verifier.SumcheckRound`."""
    return prover.StandardRound(prover.ProductSummand(degree), challenges=challenges)


def _prove(
    rnd: prover.StandardRound, factors: Sequence[Array], transcript: Transcript
) -> tuple[Array, Array]:
    """Drive `rnd` over every variable and stack its round polys into the 2-D
    proof the verifier consumes. `StandardRound.__call__` returns the raw
    round-poly array (not a `RoundMsg`), so `fold_rounds` yields a `list[Array]`;
    stacking recovers the `(rounds, degree+1)` proof shape the scan driver reads
    as `msgs.round_poly`. Returns the final folded stacked state alongside the
    proof."""
    state = fnp.stack(list(factors))
    rounds = log2_strict_usize(state.shape[-1])
    final_state, _, msgs = fold_rounds(rnd, state, transcript, rounds)
    return final_state, fnp.stack(msgs)


def _fs_point(proof: Array) -> Array:
    """The evaluation point the prover folded at, replayed from the round polys
    over a fresh sponge. `SumcheckRound` samples each challenge internally and
    does not surface it, so it is re-derived by the same Fiat-Shamir sampling the
    prover ran — identical sponge, identical messages, identical challenges."""
    transcript: Transcript = cheap_transcript(KB)
    challenges = []
    for msg in proof:
        transcript, r = transcript.observe_and_sample(msg, 1)
        challenges.append(r[0])
    return fnp.stack(challenges)


def _running(claim: Array, rounds: int = 1) -> RunningClaim:
    """A fresh replay carry for driving one round directly."""
    return RunningClaim(claim, fnp.zeros((rounds,), claim.dtype), fnp.int32(0))


class SumcheckRoundtripTest(absltest.TestCase):
    def _roundtrip(self, factors: Sequence[Array], degree: int) -> None:
        claimed = fnp.sum(product(list(factors)))
        _, proof = _prove(_standard(degree), factors, cheap_transcript(KB))
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(degree, challenges=_CH),
            claimed,
            proof,
            cheap_transcript(KB),
        )
        self.assertTrue(bool(ok))
        # Verifier rebinds the challenges from a fresh, identical sponge: its bound
        # point equals the prover's sampled challenges (Fiat-Shamir lockstep).
        self.assertTrue(bool(fnp.all(point == _fs_point(proof))))
        want = product([eval_mle_oracle(f, point) for f in factors])
        self.assertTrue(bool(final_claim == want))

    def test_degree1_single_mle_roundtrip(self) -> None:
        f = rand_field(40, (1 << 4,), KB)
        self._roundtrip((f,), degree=1)

    def test_degree2_product_roundtrip(self) -> None:
        a = rand_field(42, (1 << 4,), KB)
        b = rand_field(43, (1 << 4,), KB)
        self._roundtrip((a, b), degree=2)

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_extension_challenges_roundtrip(self) -> None:
        EF = zk_dtypes.koalabearx4_mont
        f = rand_field(52, (1 << 4,), KB).astype(EF)
        claimed = fnp.sum(f)
        _, proof = _prove(_standard(1, ChallengePolicy(EF)), [f], cheap_transcript(EF))
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(1, challenges=ChallengePolicy(EF)),
            claimed,
            proof,
            cheap_transcript(EF),
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == eval_mle_oracle(f, point)))

    def test_final_claim_matches_prover_final_state(self) -> None:
        a = rand_field(45, (1 << 3,), KB)
        b = rand_field(46, (1 << 3,), KB)
        claimed = fnp.sum(a * b)
        final_state, proof = _prove(_standard(2), [a, b], cheap_transcript(KB))
        _, final_claim, _, ok = verify(
            verifier.SumcheckRound(2, challenges=_CH),
            claimed,
            proof,
            cheap_transcript(KB),
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == product([s[0] for s in final_state])))

    def test_single_round_reduces_and_threads(self) -> None:
        # The verifier's claim reduction must equal the prover's folded sum at the
        # same challenge (the round-to-round identity). Prover and verifier each
        # drive a fresh, identical sponge and observe the same message, so they
        # sample the same challenge.
        p_round = _standard(1)
        v_round = verifier.SumcheckRound(1, challenges=_CH)
        f = rand_field(56, (8,), KB)
        state, _, msg = p_round(f[None], cheap_transcript(KB))
        state_v, _, ok = v_round(_running(msg[0] + msg[1]), cheap_transcript(KB), msg)
        next_claim = state_v.value
        self.assertTrue(bool(ok))
        self.assertTrue(bool(next_claim == fnp.sum(state[0])))

    def test_check_reduce_matches_fused_call(self) -> None:
        # The FS-decoupled seam must be the same round math as the fused hop:
        # an external observe_and_sample + check_reduce reproduces __call__'s
        # (claim, challenge, ok) exactly.
        v_round = verifier.SumcheckRound(1, challenges=_CH)
        f = rand_field(57, (8,), KB)
        _, _, msg = _standard(1)(f[None], cheap_transcript(KB))
        claim = msg[0] + msg[1]
        fused, _, fused_ok = v_round(_running(claim), cheap_transcript(KB), msg)
        fused_claim, fused_r = fused.value, fused.point[0]
        _, r = cheap_transcript(KB).observe_and_sample(msg, 1)
        split_claim, split_ok = v_round.check_reduce(claim, msg, r[0])
        self.assertTrue(bool(r[0] == fused_r))
        self.assertTrue(bool(split_claim == fused_claim))
        self.assertEqual(bool(split_ok), bool(fused_ok))

    def test_check_reduce_rejects_wrong_claim(self) -> None:
        v_round = verifier.SumcheckRound(1, challenges=_CH)
        f = rand_field(58, (8,), KB)
        _, _, msg = _standard(1)(f[None], cheap_transcript(KB))
        _, ok = v_round.check_reduce(
            msg[0] + msg[1] + fnp.array(1, KB), msg, fnp.array(3, KB)
        )
        self.assertFalse(bool(ok))

    def test_wrong_claimed_sum_rejected(self) -> None:
        f = rand_field(48, (1 << 4,), KB)
        _, proof = _prove(_standard(1), [f], cheap_transcript(KB))
        bad = fnp.sum(f) + fnp.array(1, KB)
        _, _, _, ok = verify(
            verifier.SumcheckRound(1, challenges=_CH), bad, proof, cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))

    def test_tampered_round_message_rejected(self) -> None:
        # Bumping a middle round's message breaks that round's
        # s(0)+s(1) == previous-claim link, even though round 0 still matches.
        f = rand_field(50, (1 << 4,), KB)
        claimed = fnp.sum(f)
        _, proof = _prove(_standard(1), [f], cheap_transcript(KB))
        proof = proof.at[2, 0].add(fnp.array(1, KB))
        _, _, _, ok = verify(
            verifier.SumcheckRound(1, challenges=_CH),
            claimed,
            proof,
            cheap_transcript(KB),
        )
        self.assertFalse(bool(ok))

    def test_empty_proof_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify(
                verifier.SumcheckRound(1, challenges=_CH),
                fnp.zeros((), KB),
                fnp.zeros((0, 2), KB),
                cheap_transcript(KB),
            )

    def test_non_2d_proof_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify(
                verifier.SumcheckRound(1, challenges=_CH),
                fnp.zeros((), KB),
                fnp.zeros((2,), KB),
                cheap_transcript(KB),
            )

    def test_wrong_message_width_raises(self) -> None:
        # A degree-2 verifier requires width-3 (degree+1) rounds; a width-2
        # (degree-1) proof is a malformed input, not a soundness failure.
        f = rand_field(54, (1 << 2,), KB)
        _, proof = _prove(_standard(1), [f], cheap_transcript(KB))
        with self.assertRaises(ValueError):
            verify(
                verifier.SumcheckRound(2, challenges=_CH),
                fnp.sum(f),
                proof,
                cheap_transcript(KB),
            )

    def test_degree_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            verifier.SumcheckRound(0, challenges=_CH)


class CoeffsSumcheckRoundTest(absltest.TestCase):
    def test_accepts_consistent_claim_and_reduces(self) -> None:
        coeffs = fnp.array([3, 5, 2, 7], KB)  # s(0) = 3, s(1) = 17
        claim = coeffs[0] + fnp.sum(coeffs)
        reduced, _, ok = verifier.CoeffsSumcheckRound(3, challenges=_CH)(
            _running(claim), cheap_transcript(KB), coeffs
        )
        next_claim, r = reduced.value, reduced.point[0]
        self.assertTrue(bool(ok))
        # The reduction is the coefficient evaluation at the sampled point.
        want = coeffs[0] + r * (coeffs[1] + r * (coeffs[2] + r * coeffs[3]))
        self.assertTrue(bool(next_claim == want))

    def test_wrong_claim_rejected(self) -> None:
        coeffs = fnp.array([3, 5, 2, 7], KB)
        bad = coeffs[0] + fnp.sum(coeffs) + fnp.array(1, KB)
        _, _, ok = verifier.CoeffsSumcheckRound(3, challenges=_CH)(
            _running(bad), cheap_transcript(KB), coeffs
        )
        self.assertFalse(bool(ok))

    def test_multi_limb_challenge_extends_transcript_field(self) -> None:
        EF = zk_dtypes.koalabearx4_mont
        coeffs = fnp.array([3, 5, 2, 7], KB).astype(EF)
        claim = coeffs[0] + fnp.sum(coeffs)
        reduced, _, ok = verifier.CoeffsSumcheckRound(
            3, challenges=ChallengePolicy(EF)
        )(_running(claim), cheap_transcript(KB), coeffs)
        next_claim, r = reduced.value, reduced.point[0]
        self.assertTrue(bool(ok))
        self.assertEqual(r.dtype, EF)
        self.assertEqual(next_claim.dtype, EF)

    def test_wrong_message_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            verifier.CoeffsSumcheckRound(3, challenges=_CH)(
                fnp.zeros((), KB), cheap_transcript(KB), fnp.zeros((3,), KB)
            )

    def test_validates_degree(self) -> None:
        with self.assertRaises(ValueError):
            verifier.CoeffsSumcheckRound(0, challenges=_CH)


class CompressedCoeffsRoundtripTest(absltest.TestCase):
    """The compressed [c0, c2] wire (`prover.CompressedProductRound` /
    `verifier.CompressedCoeffsSumcheckRound`). The scan driver `prove` is
    summand-based, so the rounds run in a plain per-variable loop — the way the
    Ligerito driver binds them."""

    def test_product_roundtrip(self) -> None:
        a = rand_field(60, (1 << 4,), KB)
        b = rand_field(61, (1 << 4,), KB)
        p_round = prover.CompressedProductRound(challenges=_CH)
        v_round = verifier.CompressedCoeffsSumcheckRound(challenges=_CH)

        state = fnp.stack([a, b])
        tp: Transcript = cheap_transcript(KB)
        msgs = []
        for _ in range(4):
            state, tp, msg = p_round(state, tp)
            msgs.append(msg)

        claim = fnp.sum(a * b)
        tv: Transcript = cheap_transcript(KB)
        point = []
        running = _running(claim, len(msgs))
        for msg in msgs:
            running, tv, ok = v_round(running, tv, msg)
            self.assertTrue(bool(ok))
        claim, point = running.value, list(running.point)
        # Terminal check: the reduced claim equals the product of the fully
        # folded factors — and equals the oracle eval at the bound point (the
        # Fiat-Shamir lockstep of the two fresh, identical sponges).
        self.assertTrue(bool(claim == state[0][0] * state[1][0]))
        pt = fnp.stack(point)
        want = eval_mle_oracle(a, pt) * eval_mle_oracle(b, pt)
        self.assertTrue(bool(claim == want))

    def test_check_reduce_matches_fused_call(self) -> None:
        # The FS-decoupled seam reproduces the fused hop's reduction exactly
        # (c1 reconstruction included) for the same external challenge.
        a = rand_field(66, (1 << 3,), KB)
        b = rand_field(67, (1 << 3,), KB)
        v_round = verifier.CompressedCoeffsSumcheckRound(challenges=_CH)
        _, _, msg = prover.CompressedProductRound(challenges=_CH)(
            fnp.stack([a, b]), cheap_transcript(KB)
        )
        claim = fnp.sum(a * b)
        fused, _, _ = v_round(_running(claim), cheap_transcript(KB), msg)
        fused_claim, fused_r = fused.value, fused.point[0]
        _, r = cheap_transcript(KB).observe_and_sample(msg, 1)
        split_claim, split_ok = v_round.check_reduce(claim, msg, r[0])
        self.assertTrue(bool(r[0] == fused_r))
        self.assertTrue(bool(split_claim == fused_claim))
        self.assertTrue(bool(split_ok))

    def test_tampered_message_breaks_terminal_claim(self) -> None:
        # The compressed form has no per-round redundancy (c1 comes from the
        # claim), so a tamper surfaces at the terminal check, not mid-loop.
        a = rand_field(62, (1 << 3,), KB)
        b = rand_field(63, (1 << 3,), KB)
        p_round = prover.CompressedProductRound(challenges=_CH)
        v_round = verifier.CompressedCoeffsSumcheckRound(challenges=_CH)
        state = fnp.stack([a, b])
        tp: Transcript = cheap_transcript(KB)
        msgs = []
        for _ in range(3):
            state, tp, msg = p_round(state, tp)
            msgs.append(msg)
        msgs[1] = msgs[1].at[0].add(fnp.array(1, KB))
        claim = fnp.sum(a * b)
        tv: Transcript = cheap_transcript(KB)
        running = _running(claim, len(msgs))
        for msg in msgs:
            running, tv, _ = v_round(running, tv, msg)
        claim = running.value
        self.assertFalse(bool(claim == state[0][0] * state[1][0]))

    def test_wrong_message_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            verifier.CompressedCoeffsSumcheckRound(challenges=_CH)(
                _running(fnp.zeros((), KB)), cheap_transcript(KB), fnp.zeros((3,), KB)
            )


if __name__ == "__main__":
    absltest.main()
