# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.univariate import eval_univariate
from zorch.sumcheck import prover
from zorch.sumcheck.prover import prove
from zorch.sumcheck.testing import eval_mle_oracle, product
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript

KB = zk_dtypes.koalabear_mont
_GPU_BACKEND = jax.default_backend() == "gpu"


class SumcheckProveTest(absltest.TestCase):
    def _check_identity(
        self, factors: Sequence[Array], degree: int, n: int, **prove_kwargs: Any
    ) -> prover.RoundMsg:
        """Full-domain sumcheck identity; `prove_kwargs` forward driver options
        (e.g. the extension-challenge field). Returns the stacked round messages."""
        final_state, _, msgs = prove(
            prover.SumcheckRound(degree=degree),
            list(factors),
            cheap_transcript(KB),
            **prove_kwargs,
        )
        proof = msgs.round_poly
        # The sponge derives the challenges; read them back to check the identity.
        challenges = msgs.challenge

        # claimed sum == s_0(0) + s_0(1)
        claimed = jnp.sum(product(list(factors)))
        self.assertTrue(bool(claimed == proof[0][0] + proof[0][1]))

        # round-to-round: s_i(0)+s_i(1) == s_{i-1}(r_{i-1})
        for i in range(1, n):
            lhs = proof[i][0] + proof[i][1]
            rhs = eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))

        # final fold == product of the MLEs evaluated at the challenge point
        want = product([eval_mle_oracle(f, challenges) for f in factors])
        self.assertTrue(bool(product([s[0] for s in final_state]) == want))
        last = eval_univariate(proof[n - 1], challenges[n - 1])
        self.assertTrue(bool(last == want))
        return msgs

    def _check_truncated_identity(
        self, factors: Sequence[Array], degree: int, n: int, **prove_kwargs: Any
    ) -> prover.RoundMsg:
        """`eval_start=1` sumcheck identity: each round omits s(0), so reconstruct
        s_i(0) = claim − s_i(1) and check the running claim lands on the MLE
        product. `prove_kwargs` forward driver options (e.g. EF challenges)."""
        final_state, _, msgs = prove(
            prover.SumcheckRound(degree=degree),
            list(factors),
            cheap_transcript(KB),
            eval_start=1,
            **prove_kwargs,
        )
        proof, challenges = msgs.round_poly, msgs.challenge
        self.assertEqual(proof[0].shape, (degree,))  # {s(1)..s(degree)}, s(0) gone
        claim = jnp.sum(product(list(factors)))
        for i in range(n):
            s0 = claim - proof[i][0]  # reconstruct s_i(0) = claim − s_i(1)
            full_evals = jnp.concatenate([jnp.atleast_1d(s0), proof[i]])
            claim = eval_univariate(full_evals, challenges[i])  # claim_{i+1}=s_i(r_i)
        want = product([eval_mle_oracle(f, challenges) for f in factors])
        self.assertTrue(bool(claim == want))
        self.assertTrue(bool(product([s[0] for s in final_state]) == want))
        return msgs

    def test_empty_state_raises(self) -> None:
        with self.assertRaises(ValueError):
            prove(prover.SumcheckRound(degree=1), [], cheap_transcript(KB))

    def test_degree1_single_mle(self) -> None:
        f = rand_field(20, (1 << 4,), KB)
        self._check_identity((f,), degree=1, n=4)

    def test_degree2_product_two_mles(self) -> None:
        a = rand_field(22, (1 << 4,), KB)
        b = rand_field(23, (1 << 4,), KB)
        self._check_identity((a, b), degree=2, n=4)

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_degree1_extension_challenges(self) -> None:
        EF = zk_dtypes.koalabearx4_mont
        f = rand_field(30, (1 << 4,), KB).astype(EF)
        final_state, _, msgs = prove(
            prover.SumcheckRound(degree=1), [f], cheap_transcript(EF)
        )
        proof = msgs.round_poly
        challenges = msgs.challenge  # extension-field challenges from the EF sponge
        self.assertTrue(bool(jnp.sum(f) == proof[0][0] + proof[0][1]))
        for i in range(1, 4):
            lhs = proof[i][0] + proof[i][1]
            rhs = eval_univariate(proof[i - 1], challenges[i - 1])
            self.assertTrue(bool(lhs == rhs))
        self.assertTrue(bool(final_state[0][0] == eval_mle_oracle(f, challenges)))

    def test_eval_start1_truncates_round0_values(self) -> None:
        # eval_start=1 drops s(0): round 0 (before any challenge) computes the same
        # lift as the full domain, so its message is the full round poly minus s(0).
        a = rand_field(50, (1 << 4,), KB)
        b = rand_field(51, (1 << 4,), KB)
        rnd, factors = prover.SumcheckRound(degree=2), [a, b]
        _, _, full = prove(rnd, factors, cheap_transcript(KB))
        _, _, trunc = prove(rnd, factors, cheap_transcript(KB), eval_start=1)
        self.assertEqual(full.round_poly[0].shape, (3,))  # [s(0), s(1), s(2)]
        self.assertEqual(trunc.round_poly[0].shape, (2,))  # [s(1), s(2)]
        self.assertTrue(bool(jnp.all(trunc.round_poly[0] == full.round_poly[0][1:])))

    def test_eval_start1_truncated_identity(self) -> None:
        # The compressed wire form is sound across the whole proof.
        a = rand_field(52, (1 << 4,), KB)
        b = rand_field(53, (1 << 4,), KB)
        self._check_truncated_identity((a, b), degree=2, n=4)

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_extension_challenge_from_base_transcript(self) -> None:
        # A base-field transcript with extension-field fold challenges (the SWIRL
        # stacking / zerocheck shape): each challenge is `limbs` base squeezes
        # reinterpreted as one EF element, vs the EF-sponge case above.
        EF = zk_dtypes.koalabearx4_mont
        a = rand_field(54, (1 << 4,), KB).astype(EF)
        b = rand_field(55, (1 << 4,), KB).astype(EF)
        msgs = self._check_identity(
            (a, b), degree=2, n=4, challenge_dtype=EF, challenge_limbs=4
        )
        self.assertEqual(msgs.challenge.dtype, EF)  # EF challenges off a base sponge

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_eval_start1_with_extension_challenge(self) -> None:
        # Both controls together — the exact openvm stacking shape: {s(1), s(2)}
        # round polys folded by EF challenges drawn from a base transcript.
        EF = zk_dtypes.koalabearx4_mont
        a = rand_field(56, (1 << 4,), KB).astype(EF)
        b = rand_field(57, (1 << 4,), KB).astype(EF)
        msgs = self._check_truncated_identity(
            (a, b), degree=2, n=4, challenge_dtype=EF, challenge_limbs=4
        )
        self.assertEqual(msgs.challenge.dtype, EF)

    def test_multi_limb_without_dtype_rejected(self) -> None:
        # >1 squeeze with no dtype to pack them into would advance the transcript
        # past squeezes the fold never consumes — a silent desync; reject it.
        f = rand_field(58, (1 << 3,), KB)
        with self.assertRaises(ValueError):
            prove(
                prover.SumcheckRound(degree=1),
                [f],
                cheap_transcript(KB),
                challenge_limbs=2,
            )

    def test_limbs_dtype_packing_mismatch_rejected(self) -> None:
        # 8 base squeezes reinterpret as 2 koalabearx4 elements, not 1 — the
        # surplus would desync the verifier, so fail loud (raised while tracing,
        # before any EF reduction, so this is backend-agnostic).
        EF = zk_dtypes.koalabearx4_mont
        a = rand_field(59, (1 << 3,), KB).astype(EF)
        with self.assertRaises(ValueError):
            prove(
                prover.SumcheckRound(degree=1),
                [a],
                cheap_transcript(KB),
                challenge_dtype=EF,
                challenge_limbs=8,
            )


if __name__ == "__main__":
    absltest.main()
