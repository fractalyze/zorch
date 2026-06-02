# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.prove import prove
from zorch.sumcheck import ProductSumcheckRound, SumcheckVerifier
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript
from zorch.verify import verify

KB = zk_dtypes.koalabear


def _product(factors):
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _eval_mle(evals, point):
    """Oracle: evaluate a multilinear (2^n evals) at `point` (MSB-first)."""
    cur = evals
    for r in point:
        half = cur.shape[-1] // 2
        cur = cur[..., :half] + r * (cur[..., half:] - cur[..., :half])
    return cur[0]


class SumcheckVerifyTest(absltest.TestCase):
    def _roundtrip(self, factors, degree, n, seed):
        challenges = rand_field(seed, (n,), KB)
        claimed = jnp.sum(_product(list(factors)))
        _, _, proof = prove(
            ProductSumcheckRound(degree), list(factors), StubTranscript(challenges)
        )
        point, final_claim, _, ok = verify(
            SumcheckVerifier(degree), claimed, proof, StubTranscript(challenges)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.all(point == challenges)))
        want = _product([_eval_mle(f, point) for f in factors])
        self.assertTrue(bool(final_claim == want))

    def test_degree1_single_mle_roundtrip(self):
        f = rand_field(40, (1 << 4,), KB)
        self._roundtrip((f,), degree=1, n=4, seed=41)

    def test_degree2_product_roundtrip(self):
        a = rand_field(42, (1 << 4,), KB)
        b = rand_field(43, (1 << 4,), KB)
        self._roundtrip((a, b), degree=2, n=4, seed=44)

    def test_extension_challenges_roundtrip(self):
        EF = zk_dtypes.koalabearx4
        f = rand_field(52, (1 << 4,), KB).astype(EF)
        challenges = rand_field(53, (4,), KB).astype(EF)
        claimed = jnp.sum(f)
        _, _, proof = prove(ProductSumcheckRound(1), [f], StubTranscript(challenges))
        point, final_claim, _, ok = verify(
            SumcheckVerifier(1), claimed, proof, StubTranscript(challenges)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == _eval_mle(f, point)))

    def test_final_claim_matches_prover_final_state(self):
        a = rand_field(45, (1 << 3,), KB)
        b = rand_field(46, (1 << 3,), KB)
        challenges = rand_field(47, (3,), KB)
        claimed = jnp.sum(a * b)
        final_state, _, proof = prove(
            ProductSumcheckRound(2), [a, b], StubTranscript(challenges)
        )
        _, final_claim, _, ok = verify(
            SumcheckVerifier(2), claimed, proof, StubTranscript(challenges)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == _product([s[0] for s in final_state])))

    def test_single_round_reduces_and_threads(self):
        # The verifier's claim reduction must equal the prover's folded sum at
        # the same challenge (the sumcheck round-to-round identity).
        prover, verifier = ProductSumcheckRound(1), SumcheckVerifier(1)
        f = rand_field(56, (8,), KB)
        state, _, msg = prover([f], StubTranscript(jnp.array([5, 0], KB)))
        next_claim, t2, r, ok = verifier(
            msg[0] + msg[1], msg, StubTranscript(jnp.array([5, 0], KB))
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(r == jnp.array(5, KB)))
        self.assertEqual(t2.pos, 1)
        self.assertTrue(bool(next_claim == jnp.sum(state[0])))

    def test_wrong_claimed_sum_rejected(self):
        f = rand_field(48, (1 << 4,), KB)
        challenges = rand_field(49, (4,), KB)
        _, _, proof = prove(ProductSumcheckRound(1), [f], StubTranscript(challenges))
        bad = jnp.sum(f) + jnp.array(1, KB)
        _, _, _, ok = verify(
            SumcheckVerifier(1), bad, proof, StubTranscript(challenges)
        )
        self.assertFalse(bool(ok))

    def test_tampered_round_message_rejected(self):
        # Bumping a middle round's message breaks that round's
        # s(0)+s(1) == previous-claim link, even though round 0 still matches.
        f = rand_field(50, (1 << 4,), KB)
        challenges = rand_field(51, (4,), KB)
        claimed = jnp.sum(f)
        _, _, proof = prove(ProductSumcheckRound(1), [f], StubTranscript(challenges))
        proof = proof.at[2, 0].add(jnp.array(1, KB))
        _, _, _, ok = verify(
            SumcheckVerifier(1), claimed, proof, StubTranscript(challenges)
        )
        self.assertFalse(bool(ok))

    def test_empty_proof_raises(self):
        with self.assertRaises(ValueError):
            verify(
                SumcheckVerifier(1),
                jnp.zeros((), KB),
                jnp.zeros((0, 2), KB),
                StubTranscript(jnp.zeros(1, KB)),
            )

    def test_non_2d_proof_raises(self):
        with self.assertRaises(ValueError):
            verify(
                SumcheckVerifier(1),
                jnp.zeros((), KB),
                jnp.zeros((2,), KB),
                StubTranscript(jnp.zeros(1, KB)),
            )

    def test_wrong_message_width_raises(self):
        # A degree-2 verifier requires width-3 (degree+1) rounds; a width-2
        # (degree-1) proof is a malformed input, not a soundness failure.
        f = rand_field(54, (1 << 2,), KB)
        challenges = rand_field(55, (2,), KB)
        _, _, proof = prove(ProductSumcheckRound(1), [f], StubTranscript(challenges))
        with self.assertRaises(ValueError):
            verify(SumcheckVerifier(2), jnp.sum(f), proof, StubTranscript(challenges))

    def test_degree_must_be_positive(self):
        with self.assertRaises(ValueError):
            SumcheckVerifier(0)


if __name__ == "__main__":
    absltest.main()
