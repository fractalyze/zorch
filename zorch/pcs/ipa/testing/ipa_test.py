# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.pcs.ipa.challenger import TranscriptChallenger
from zorch.pcs.ipa.config import IpaProof
from zorch.pcs.ipa.math import (
    _check_pow2,
    challenge_vector,
    eval_challenge_poly,
    inner_powers,
)
from zorch.pcs.ipa.prover import IpaProver, IpaProverData
from zorch.pcs.ipa.testing import basis
from zorch.pcs.ipa.verifier import IpaVerifier, reduce_opening, settle
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

# The fold/challenge math is field-agnostic; exercise it over a small base field
# (CPU-friendly), independent of bn254 and the GPU msm path — the split KZG draws
# with `_quotient_and_eval`.
KB = zk_dtypes.koalabear_mont


class ChallengeMathTest(absltest.TestCase):
    def test_inner_powers(self) -> None:
        x = jnp.array(3, dtype=KB)
        self.assertEqual([int(v) for v in inner_powers(x, 4)], [1, 3, 9, 27])

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
        self.assertEqual(int(folded[0]), int(jnp.sum(challenge_vector(u, u_inv) * v)))


# --- full commit -> open -> verify (GPU: lax.msm is GPU-only) -------------------
#
# The fold / MSM path is curve-generic — only the bases' dtype changes — so run it
# over every G1 the seam targets. Scalars are drawn in Montgomery form (the
# production encoding) over the standard-domain toy basis, the same split KZG's
# round-trip test uses.

_GPU = jax.default_backend() == "gpu"

# (subtest name, Montgomery scalar field, standard-domain toy curve)
_CURVES = (
    ("bn254", zk_dtypes.bn254_sf_mont, basis.BN254),
    ("pallas", zk_dtypes.pallas_sf_mont, basis.PALLAS),
)


def _transcript(sf: type) -> DuplexTranscript:
    return cheap_transcript(sf)


@absltest.skipUnless(_GPU, "IPA commit/open/verify use lax.msm, a GPU-only kernel")
class IpaRoundTripTest(parameterized.TestCase):
    def _commit_open(
        self, sf: type, curve: basis.ToyCurve
    ) -> tuple[IpaVerifier, Array, Array, Array, list[IpaProof]]:
        key = basis.toy_key(curve, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=sf)  # p(x) = 3 + x + 4x² + x³
        x = jnp.array(7, dtype=sf)
        commitment, data = IpaProver(key).commit([coeffs])
        values, proof, _ = IpaProver(key).open(data, [x], _transcript(sf))
        return IpaVerifier(key), x, commitment, values, proof

    @parameterized.named_parameters(*_CURVES)
    def test_value_is_evaluation(self, sf: type, curve: basis.ToyCurve) -> None:
        # p(7) = 3 + 7 + 4·49 + 343 = 549.
        _, _, _, values, _ = self._commit_open(sf, curve)
        self.assertEqual(int(values[0]), 549)

    @parameterized.named_parameters(*_CURVES)
    def test_open_verifies(self, sf: type, curve: basis.ToyCurve) -> None:
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        ok, _ = verifier.verify(commitment, [x], values, proof, _transcript(sf))
        self.assertTrue(bool(ok))

    @parameterized.named_parameters(*_CURVES)
    def test_wrong_value_rejected(self, sf: type, curve: basis.ToyCurve) -> None:
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = values + jnp.array(1, dtype=sf)
        ok, _ = verifier.verify(commitment, [x], bad, proof, _transcript(sf))
        self.assertFalse(bool(ok))

    @parameterized.named_parameters(*_CURVES)
    def test_reduced_claim_defers_the_msm(
        self, sf: type, curve: basis.ToyCurve
    ) -> None:
        # reduce_opening, driven by an injected IpaChallenger, reaches the same
        # accept verdict as verify once its deferred claim is settled — the
        # accumulation reuse contract (an arkworks consumer swaps in its own
        # challenger here).
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        _, claim = reduce_opening(
            verifier.key,
            commitment[0],
            x,
            values[0],
            proof[0],
            TranscriptChallenger(_transcript(sf), sf),
        )
        self.assertTrue(bool(settle(verifier.key, claim)))

    @parameterized.named_parameters(*_CURVES)
    def test_wrong_commitment_rejected(self, sf: type, curve: basis.ToyCurve) -> None:
        # The Fiat-Shamir now binds the commitment: verifying against a different
        # one rejects (statement binding the bare fold lacked).
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = jnp.stack([verifier.key.u])  # U as a stand-in P ≠ the real commitment
        ok, _ = verifier.verify(bad, [x], values, proof, _transcript(sf))
        self.assertFalse(bool(ok))


class IpaBatchValidationTest(absltest.TestCase):
    # The batch-length guards fire before any MSM, so these need no GPU.
    def test_open_rejects_batch_mismatch(self) -> None:
        sf = zk_dtypes.bn254_sf_mont
        key = basis.toy_key(basis.BN254, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=sf)
        x = jnp.array(7, dtype=sf)
        # commitments is never read — open raises on the length check first.
        dummy = jnp.zeros((1,), dtype=basis.BN254.g1)
        with self.assertRaises(ValueError):
            IpaProver(key).open(
                IpaProverData((coeffs,), dummy), [x, x], _transcript(sf)
            )


if __name__ == "__main__":
    absltest.main()
