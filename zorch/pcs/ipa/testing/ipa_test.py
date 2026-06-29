# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.pcs.ipa.config import IpaProof
from zorch.pcs.ipa.math import (
    _check_pow2,
    challenge_vector,
    eval_challenge_poly,
    inner_powers,
)
from zorch.pcs.ipa.prover import IpaProver, IpaProverData
from zorch.pcs.ipa.testing.basis import toy_key
from zorch.pcs.ipa.verifier import IpaVerifier, reduce_opening
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

# The fold/challenge math is field-agnostic; exercise it over a small base field
# (CPU-friendly), independent of bn254 and the GPU msm path — the split KZG draws
# with `_quotient_and_eval`.
KB = zk_dtypes.koalabear_mont


class ChallengeMathTest(absltest.TestCase):
    def test_inner_powers(self) -> None:
        x = jnp.array(3, dtype=KB)
        self.assertEqual(
            [int(v) for v in inner_powers(x, 4)], [1, 3, 9, 27]
        )

    def test_check_pow2_rejects_non_power_of_two(self) -> None:
        for bad in (0, 3, 6, 12):
            with self.assertRaises(ValueError):
                _check_pow2(bad)

    def test_challenge_poly_matches_explicit_s(self) -> None:
        # eval_challenge_poly(u, x) is the succinct read of ⟨challenge_vector, b⟩.
        u = jnp.array([2, 3, 5], dtype=KB)  # k = 3 → n = 8
        u_inv = jnp.array(1, dtype=KB) / u
        x = jnp.array(7, dtype=KB)
        s = challenge_vector(u, u_inv)
        self.assertEqual(s.shape, (8,))
        explicit = jnp.sum(s * inner_powers(x, 8))
        self.assertEqual(int(eval_challenge_poly(u, u_inv, x)), int(explicit))

    def test_challenge_vector_is_the_fold_inverse(self) -> None:
        # Folding any vector by the prover's basis recurrence
        # (V ← V_lo·u⁻¹ + V_hi·u) collapses to ⟨s, V⟩ — the property that makes
        # G_final = ⟨s, G⟩ reproduce the prover's folded basis.
        u = jnp.array([2, 3, 5], dtype=KB)
        u_inv = jnp.array(1, dtype=KB) / u
        v = jnp.arange(1, 9, dtype=KB)
        folded = v
        for j in range(3):
            m = folded.shape[0] // 2
            folded = folded[:m] * u_inv[j] + folded[m:] * u[j]
        self.assertEqual(folded.shape, (1,))
        self.assertEqual(
            int(folded[0]), int(jnp.sum(challenge_vector(u, u_inv) * v))
        )


# --- full commit -> open -> verify over bn254 (GPU: lax.msm is GPU-only) --------

SF = zk_dtypes.bn254_sf
_GPU = jax.default_backend() == "gpu"


def _transcript() -> DuplexTranscript:
    return cheap_transcript(SF)


@absltest.skipUnless(_GPU, "IPA commit/open/verify use lax.msm, a GPU-only kernel")
class IpaRoundTripTest(absltest.TestCase):
    def setUp(self) -> None:
        self.key = toy_key(n=4)
        self.coeffs = jnp.array([3, 1, 4, 1], dtype=SF)  # p(x) = 3 + x + 4x² + x³
        self.x = jnp.array(7, dtype=SF)
        self.prover = IpaProver(self.key)
        self.verifier = IpaVerifier(self.key)

    def _open(self) -> tuple[Array, Array, list[IpaProof]]:
        commitment, data = self.prover.commit([self.coeffs])
        values, proof, _ = self.prover.open(data, [self.x], _transcript())
        return commitment, values, proof

    def test_value_is_evaluation(self) -> None:
        # p(7) = 3 + 7 + 4·49 + 343 = 549.
        _, values, _ = self._open()
        self.assertEqual(int(values[0]), 549)

    def test_open_verifies(self) -> None:
        commitment, values, proof = self._open()
        ok, _ = self.verifier.verify(
            commitment, [self.x], values, proof, _transcript()
        )
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        commitment, values, proof = self._open()
        bad = values + jnp.array(1, dtype=SF)
        ok, _ = self.verifier.verify(
            commitment, [self.x], bad, proof, _transcript()
        )
        self.assertFalse(bool(ok))

    def test_reduced_claim_defers_the_msm(self) -> None:
        # reduce_opening reaches the same accept verdict as verify when its
        # deferred claim is settled — the accumulation reuse contract.
        commitment, values, proof = self._open()
        from zorch.pcs.ipa.verifier import settle

        _, claim = reduce_opening(
            self.key, commitment[0], self.x, values[0], proof[0], _transcript()
        )
        self.assertTrue(bool(settle(self.key, claim)))


class IpaBatchValidationTest(absltest.TestCase):
    # The batch-length guards fire before any MSM, so these need no GPU.
    def test_open_rejects_batch_mismatch(self) -> None:
        key = toy_key(n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=SF)
        x = jnp.array(7, dtype=SF)
        with self.assertRaises(ValueError):
            IpaProver(key).open(IpaProverData((coeffs,)), [x, x], _transcript())


if __name__ == "__main__":
    absltest.main()
