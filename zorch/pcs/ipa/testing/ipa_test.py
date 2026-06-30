# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.pcs.ipa.challenger import TranscriptChallenger
from zorch.pcs.ipa.config import IpaProof, IpaZkProof
from zorch.pcs.ipa.math import challenge_vector, eval_challenge_poly
from zorch.pcs.ipa.prover import IpaProver, IpaProverData, _open_one_zk
from zorch.pcs.ipa.setup import IpaKey
from zorch.pcs.ipa.testing import basis
from zorch.pcs.ipa.verifier import (
    IpaVerifier,
    reduce_opening,
    reduce_opening_zk,
    settle,
)
from zorch.pcs.testing import curves
from zorch.poly.univariate import powers
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

# The check-poly math touches only the scalar field, so exercise it over the
# curve's scalar field — Pallas Fr, the field the IPA fold actually runs in and
# the one #339 byte-matches arkworks over — not the curve's EC / GPU msm path
# (the same field-only / msm split KZG draws with `_quotient_and_eval`).
SF = zk_dtypes.pallas_sf_mont


class ChallengeMathTest(absltest.TestCase):
    def test_challenge_poly_matches_explicit_s(self) -> None:
        # eval_challenge_poly(u, x) is the succinct read of ⟨challenge_vector, b⟩.
        u = jnp.array([2, 3, 5], dtype=SF)  # k = 3 → n = 8
        x = jnp.array(7, dtype=SF)
        s = challenge_vector(u)
        self.assertEqual(s.shape, (8,))
        explicit = jnp.sum(s * powers(x, 8))
        self.assertEqual(int(eval_challenge_poly(u, x)), int(explicit))

    def test_challenge_vector_is_the_fold_inverse(self) -> None:
        # Folding any vector by the prover's basis recurrence
        # (V ← V_lo + V_hi·u) collapses to ⟨s, V⟩ — the property that makes
        # G_final = ⟨s, G⟩ reproduce the prover's folded basis.
        u = jnp.array([2, 3, 5], dtype=SF)
        v = jnp.arange(1, 9, dtype=SF)
        folded = v
        for j in range(3):
            m = folded.shape[0] // 2
            folded = folded[:m] + folded[m:] * u[j]
        self.assertEqual(folded.shape, (1,))
        self.assertEqual(int(folded[0]), int(jnp.sum(challenge_vector(u) * v)))

    def test_challenge_vector_matches_arkworks_h_coeffs(self) -> None:
        # The dense coeffs are the check polynomial's coeffs:
        # h(X) = ∏(1 + u_j·X^{2^{k-1-j}}) (no inverses) — the descending-block
        # layout the decider's final-key MSM byte-matches arkworks ipa_pc over
        # (zorch#339 W4; the formula is the contract). For k=2,
        # h(X) = (1 + u0·X²)(1 + u1·X) → [1, u1, u0, u0·u1].
        u = jnp.array([2, 3], dtype=SF)
        self.assertEqual([int(c) for c in challenge_vector(u)], [1, 3, 2, 6])


# --- full commit -> open -> verify (GPU: lax.msm is GPU-only) -------------------
#
# The fold / MSM path is curve-generic — only the bases' dtype changes — so run it
# over every G1 the seam targets. Scalars are drawn in Montgomery form (the
# production encoding) over the standard-domain toy basis, the same split KZG's
# round-trip test uses.

_GPU = jax.default_backend() == "gpu"

# (subtest name, Montgomery scalar field, standard-domain toy curve)
_CURVES = (
    ("bn254", zk_dtypes.bn254_sf_mont, curves.BN254),
    ("pallas", zk_dtypes.pallas_sf_mont, curves.PALLAS),
)


def _transcript(sf: type) -> DuplexTranscript:
    return cheap_transcript(sf)


@absltest.skipUnless(_GPU, "IPA commit/open/verify use lax.msm, a GPU-only kernel")
class IpaRoundTripTest(parameterized.TestCase):
    def _commit_open(
        self, sf: type, curve: curves.Curve
    ) -> tuple[IpaVerifier, Array, Array, Array, list[IpaProof]]:
        key = basis.toy_key(curve, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=sf)  # p(x) = 3 + x + 4x² + x³
        x = jnp.array(7, dtype=sf)
        commitment, data = IpaProver(key).commit([coeffs])
        values, proof, _ = IpaProver(key).open(data, [x], _transcript(sf))
        return IpaVerifier(key), x, commitment, values, proof

    @parameterized.named_parameters(*_CURVES)
    def test_value_is_evaluation(self, sf: type, curve: curves.Curve) -> None:
        # p(7) = 3 + 7 + 4·49 + 343 = 549.
        _, _, _, values, _ = self._commit_open(sf, curve)
        self.assertEqual(int(values[0]), 549)

    @parameterized.named_parameters(*_CURVES)
    def test_open_verifies(self, sf: type, curve: curves.Curve) -> None:
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        ok, _ = verifier.verify(commitment, [x], values, proof, _transcript(sf))
        self.assertTrue(bool(ok))

    @parameterized.named_parameters(*_CURVES)
    def test_wrong_value_rejected(self, sf: type, curve: curves.Curve) -> None:
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = values + jnp.array(1, dtype=sf)
        ok, _ = verifier.verify(commitment, [x], bad, proof, _transcript(sf))
        self.assertFalse(bool(ok))

    @parameterized.named_parameters(*_CURVES)
    def test_reduced_claim_defers_the_msm(self, sf: type, curve: curves.Curve) -> None:
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
    def test_wrong_commitment_rejected(self, sf: type, curve: curves.Curve) -> None:
        # The Fiat-Shamir now binds the commitment: verifying against a different
        # one rejects (statement binding the bare fold lacked).
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = jnp.stack([verifier.key.u])  # U as a stand-in P ≠ the real commitment
        ok, _ = verifier.verify(bad, [x], values, proof, _transcript(sf))
        self.assertFalse(bool(ok))

    # --- hiding / zk path ---------------------------------------------------
    #
    # The zk open blinds (commitment, coeffs) with a shifted blinding poly under
    # one extra challenge, then runs the *same* fold; the verifier re-derives the
    # blinded statement and settles by the identical reduced claim. The blinding
    # preserves the opened value.

    def _commit_open_zk(
        self, sf: type, curve: curves.Curve
    ) -> tuple[IpaKey, Array, Array, Array, IpaZkProof]:
        key = basis.toy_key(curve, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=sf)  # p(x) = 3 + x + 4x² + x³
        x = jnp.array(7, dtype=sf)
        hiding = jnp.array([2, 9, 1, 8], dtype=sf)  # blinding polynomial
        hiding_rand = jnp.array(5, dtype=sf)
        commitment_randomness = jnp.array(11, dtype=sf)
        # The zk path opens a *hiding* commitment ⟨coeffs,G⟩ + cr·s (open_zk
        # removes all randomness inside the fold).
        commitment, _ = IpaProver(key).commit_zk([coeffs], [commitment_randomness])
        fs = TranscriptChallenger(_transcript(sf), sf)
        _, value, proof = _open_one_zk(
            key,
            commitment[0],
            coeffs,
            x,
            hiding,
            hiding_rand,
            commitment_randomness,
            fs,
        )
        return key, x, commitment, value, proof

    @parameterized.named_parameters(*_CURVES)
    def test_zk_value_is_evaluation(self, sf: type, curve: curves.Curve) -> None:
        # Blinding preserves the value: p(7) = 549 despite the random blinding poly.
        _, _, _, value, _ = self._commit_open_zk(sf, curve)
        self.assertEqual(int(value), 549)

    @parameterized.named_parameters(*_CURVES)
    def test_zk_open_verifies(self, sf: type, curve: curves.Curve) -> None:
        key, x, commitment, value, proof = self._commit_open_zk(sf, curve)
        _, claim = reduce_opening_zk(
            key,
            commitment[0],
            x,
            value,
            proof,
            TranscriptChallenger(_transcript(sf), sf),
        )
        self.assertTrue(bool(settle(key, claim)))

    @parameterized.named_parameters(*_CURVES)
    def test_zk_wrong_value_rejected(self, sf: type, curve: curves.Curve) -> None:
        key, x, commitment, value, proof = self._commit_open_zk(sf, curve)
        bad = value + jnp.array(1, dtype=sf)
        _, claim = reduce_opening_zk(
            key,
            commitment[0],
            x,
            bad,
            proof,
            TranscriptChallenger(_transcript(sf), sf),
        )
        self.assertFalse(bool(settle(key, claim)))


class IpaBatchValidationTest(absltest.TestCase):
    # The batch-length guards fire before any MSM, so these need no GPU.
    def test_open_rejects_batch_mismatch(self) -> None:
        sf = zk_dtypes.bn254_sf_mont
        key = basis.toy_key(curves.BN254, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=sf)
        x = jnp.array(7, dtype=sf)
        # commitments is never read — open raises on the length check first.
        dummy = jnp.zeros((1,), dtype=curves.BN254.g1)
        with self.assertRaises(ValueError):
            IpaProver(key).open(
                IpaProverData((coeffs,), dummy), [x, x], _transcript(sf)
            )


if __name__ == "__main__":
    absltest.main()
