# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.prove import prove
from zorch.sumcheck import prover, verifier
from zorch.sumcheck.testing import eval_mle_oracle, product
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.verify import verify

KB = zk_dtypes.koalabear
_GPU_BACKEND = jax.default_backend() == "gpu"


class SumcheckRoundtripTest(absltest.TestCase):
    def _roundtrip(self, factors: Sequence[Array], degree: int) -> None:
        claimed = jnp.sum(product(list(factors)))
        _, _, msgs = prove(
            prover.SumcheckRound(degree), list(factors), cheap_transcript(KB)
        )
        proof = msgs.round_poly
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(degree), claimed, proof, cheap_transcript(KB)
        )
        self.assertTrue(bool(ok))
        # Verifier rebinds the challenges from a fresh, identical sponge: its bound
        # point equals the prover's sampled challenges (Fiat-Shamir lockstep).
        self.assertTrue(bool(jnp.all(point == msgs.challenge)))
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
        EF = zk_dtypes.koalabearx4
        f = rand_field(52, (1 << 4,), KB).astype(EF)
        claimed = jnp.sum(f)
        _, _, msgs = prove(prover.SumcheckRound(1), [f], cheap_transcript(EF))
        proof = msgs.round_poly
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(1), claimed, proof, cheap_transcript(EF)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == eval_mle_oracle(f, point)))

    def test_final_claim_matches_prover_final_state(self) -> None:
        a = rand_field(45, (1 << 3,), KB)
        b = rand_field(46, (1 << 3,), KB)
        claimed = jnp.sum(a * b)
        final_state, _, msgs = prove(
            prover.SumcheckRound(2), [a, b], cheap_transcript(KB)
        )
        proof = msgs.round_poly
        _, final_claim, _, ok = verify(
            verifier.SumcheckRound(2), claimed, proof, cheap_transcript(KB)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(final_claim == product([s[0] for s in final_state])))

    def test_single_round_reduces_and_threads(self) -> None:
        # The verifier's claim reduction must equal the prover's folded sum at the
        # same challenge (the round-to-round identity). Prover and verifier each
        # drive a fresh, identical sponge and observe the same message, so they
        # sample the same challenge.
        p_round = prover.SumcheckRound(1)
        v_round = verifier.SumcheckRound(1)
        f = rand_field(56, (8,), KB)
        state, _, msg = p_round([f], cheap_transcript(KB))
        next_claim, _, _, ok = v_round(msg[0] + msg[1], msg, cheap_transcript(KB))
        self.assertTrue(bool(ok))
        self.assertTrue(bool(next_claim == jnp.sum(state[0])))

    def test_wrong_claimed_sum_rejected(self) -> None:
        f = rand_field(48, (1 << 4,), KB)
        _, _, msgs = prove(prover.SumcheckRound(1), [f], cheap_transcript(KB))
        proof = msgs.round_poly
        bad = jnp.sum(f) + jnp.array(1, KB)
        _, _, _, ok = verify(
            verifier.SumcheckRound(1), bad, proof, cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))

    def test_tampered_round_message_rejected(self) -> None:
        # Bumping a middle round's message breaks that round's
        # s(0)+s(1) == previous-claim link, even though round 0 still matches.
        f = rand_field(50, (1 << 4,), KB)
        claimed = jnp.sum(f)
        _, _, msgs = prove(prover.SumcheckRound(1), [f], cheap_transcript(KB))
        proof = msgs.round_poly
        proof = proof.at[2, 0].add(jnp.array(1, KB))
        _, _, _, ok = verify(
            verifier.SumcheckRound(1), claimed, proof, cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))

    def test_empty_proof_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify(
                verifier.SumcheckRound(1),
                jnp.zeros((), KB),
                jnp.zeros((0, 2), KB),
                cheap_transcript(KB),
            )

    def test_non_2d_proof_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify(
                verifier.SumcheckRound(1),
                jnp.zeros((), KB),
                jnp.zeros((2,), KB),
                cheap_transcript(KB),
            )

    def test_wrong_message_width_raises(self) -> None:
        # A degree-2 verifier requires width-3 (degree+1) rounds; a width-2
        # (degree-1) proof is a malformed input, not a soundness failure.
        f = rand_field(54, (1 << 2,), KB)
        _, _, msgs = prove(prover.SumcheckRound(1), [f], cheap_transcript(KB))
        proof = msgs.round_poly
        with self.assertRaises(ValueError):
            verify(verifier.SumcheckRound(2), jnp.sum(f), proof, cheap_transcript(KB))

    def test_degree_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            verifier.SumcheckRound(0)


if __name__ == "__main__":
    absltest.main()
