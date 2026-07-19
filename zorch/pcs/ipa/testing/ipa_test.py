# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import Array, lax

from zorch.pcs.ipa.challenger import TranscriptChallenger
from zorch.pcs.ipa.config import IpaProof, IpaZkProof
from zorch.pcs.ipa.math import challenge_vector, eval_challenge_poly
from zorch.pcs.ipa.prover import IpaProver, IpaProverData, _open_one, _open_one_zk
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
        u = fnp.array([2, 3, 5], dtype=SF)  # k = 3 → n = 8
        x = fnp.array(7, dtype=SF)
        s = challenge_vector(u)
        self.assertEqual(s.shape, (8,))
        explicit = fnp.sum(s * powers(x, 8))
        self.assertEqual(int(eval_challenge_poly(u, x)), int(explicit))

    def test_challenge_vector_is_the_fold_inverse(self) -> None:
        # Folding any vector by the prover's basis recurrence
        # (V ← V_lo + V_hi·u) collapses to ⟨s, V⟩ — the property that makes
        # G_final = ⟨s, G⟩ reproduce the prover's folded basis.
        u = fnp.array([2, 3, 5], dtype=SF)
        v = fnp.arange(1, 9, dtype=SF)
        folded = v
        for j in range(3):
            m = folded.shape[0] // 2
            folded = folded[:m] + folded[m:] * u[j]
        self.assertEqual(folded.shape, (1,))
        self.assertEqual(int(folded[0]), int(fnp.sum(challenge_vector(u) * v)))

    def test_challenge_vector_matches_arkworks_h_coeffs(self) -> None:
        # The dense coeffs are the check polynomial's coeffs:
        # h(X) = ∏(1 + u_j·X^{2^{k-1-j}}) (no inverses) — the descending-block
        # layout the decider's final-key MSM byte-matches arkworks ipa_pc over
        # (zorch#339 W4; the formula is the contract). For k=2,
        # h(X) = (1 + u0·X²)(1 + u1·X) → [1, u1, u0, u0·u1].
        u = fnp.array([2, 3], dtype=SF)
        self.assertEqual([int(c) for c in challenge_vector(u)], [1, 3, 2, 6])


class TranscriptChallengerPytreeTest(absltest.TestCase):
    """`TranscriptChallenger` is a registered pytree so the prover fold carries it
    through its `lax.scan` (`_open_one`): the wrapped transcript's `DuplexState`
    buffers are the leaves, `dtype` is static meta."""

    def test_flatten_roundtrip(self) -> None:
        ch = TranscriptChallenger(cheap_transcript(SF), SF)
        leaves, treedef = frx.tree_util.tree_flatten(ch)
        self.assertEqual(len(leaves), 5)  # DuplexState's 5 buffers; dtype is not a leaf
        back = frx.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(back.dtype, ch.dtype)  # the static meta survives unflatten

    def test_two_instances_share_one_treedef(self) -> None:
        # `dtype` is an object-typed meta field, so independently built challengers
        # must share one treedef — else the fold's scan zone re-traces every call
        # (conventions.md / issue #163).
        a = TranscriptChallenger(cheap_transcript(SF), SF)
        b = TranscriptChallenger(cheap_transcript(SF), SF)
        self.assertEqual(
            frx.tree_util.tree_structure(a), frx.tree_util.tree_structure(b)
        )

    def test_threads_through_jit_as_argument(self) -> None:
        ch = TranscriptChallenger(cheap_transcript(SF), SF)
        lhs, rhs = fnp.array(3, SF), fnp.array(5, SF)
        got = frx.jit(lambda c: c.challenge(lhs, rhs)[1])(ch)
        self.assertTrue(bool(got == ch.challenge(lhs, rhs)[1]))


# --- full commit -> open -> verify (GPU: lax.msm is GPU-only) -------------------
#
# The fold / MSM path is curve-generic — only the bases' dtype changes — so run it
# over every G1 the seam targets. Scalars are drawn in Montgomery form (the
# production encoding) over the standard-domain toy basis, the same split KZG's
# round-trip test uses.

_GPU = frx.default_backend() == "gpu"

# (subtest name, Montgomery scalar field, standard-domain toy curve)
_CURVES = (
    ("bn254", zk_dtypes.bn254_sf_mont, curves.BN254),
    ("pallas", zk_dtypes.pallas_sf_mont, curves.PALLAS),
)

# The zk/hiding path makes the both-Pasta byte-exactness claim, so its round-trip
# also runs over Vesta (bn254 + Pallas already cover the curve-generic fold for the
# transparent tests).
_ZK_CURVES = (*_CURVES, ("vesta", zk_dtypes.vesta_sf_mont, curves.VESTA))

# Sizes that exercise the fold's `lax.scan` at several round counts k = log₂ n
# (one binary per n; n=2 is the single-round edge), the recompile-free-compile
# property zorch#344 lands the scan for.
_CURVE_SIZES = tuple(
    (f"{name}_n{n}", sf, curve, n)
    for (name, sf, curve) in _CURVES
    for n in (2, 4, 8, 16)
)


def _transcript(sf: type) -> DuplexTranscript:
    return cheap_transcript(sf)


@absltest.skipUnless(_GPU, "IPA commit/open/verify use lax.msm, a GPU-only kernel")
class IpaRoundTripTest(parameterized.TestCase):
    def _commit_open(
        self, sf: type, curve: curves.Curve
    ) -> tuple[IpaVerifier, Array, Array, Array, list[IpaProof]]:
        key = basis.toy_key(curve, n=4)
        coeffs = fnp.array([3, 1, 4, 1], dtype=sf)  # p(x) = 3 + x + 4x² + x³
        x = fnp.array(7, dtype=sf)
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

    @parameterized.named_parameters(*_CURVE_SIZES)
    def test_open_verifies_across_sizes(
        self, sf: type, curve: curves.Curve, n: int
    ) -> None:
        # The fold's `lax.scan` runs k = log₂ n rounds; verify the commit -> open ->
        # verify round trip closes at each size (and the single-round n=2 edge), so
        # the scan fold byte-matches the shrinking fold it replaced (zorch#344).
        key = basis.toy_key(curve, n=n)
        coeffs = fnp.arange(1, n + 1, dtype=sf)
        x = fnp.array(7, dtype=sf)
        commitment, data = IpaProver(key).commit([coeffs])
        values, proof, _ = IpaProver(key).open(data, [x], _transcript(sf))
        ok, _ = IpaVerifier(key).verify(commitment, [x], values, proof, _transcript(sf))
        self.assertTrue(bool(ok))

    @parameterized.named_parameters(*_CURVES)
    def test_wrong_value_rejected(self, sf: type, curve: curves.Curve) -> None:
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = values + fnp.array(1, dtype=sf)
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
    def test_open_returns_settles_final_comm_key(
        self, sf: type, curve: curves.Curve
    ) -> None:
        # #371: _open_one returns the fully-folded generator g[0]. It must equal the
        # G_final = ⟨challenge_vector(u), G⟩ the verifier recomputes in `settle`, so an
        # accumulation consumer reads it here instead of replaying the round
        # challenges and paying a second size-n MSM.
        key = basis.toy_key(curve, n=4)
        coeffs = fnp.array([3, 1, 4, 1], dtype=sf)
        x = fnp.array(7, dtype=sf)
        commitment, _ = IpaProver(key).commit([coeffs])
        _, value, proof, final_comm_key = _open_one(
            key, commitment[0], coeffs, x, TranscriptChallenger(_transcript(sf), sf)
        )
        # The verifier's independent G_final, exactly as `settle` recomputes it.
        _, claim = reduce_opening(
            key,
            commitment[0],
            x,
            value,
            proof,
            TranscriptChallenger(_transcript(sf), sf),
        )
        s = challenge_vector(claim.u)
        g_final = lax.msm(s, key.basis[: s.shape[0]])
        aff = key.basis.dtype
        self.assertTrue(
            bool(
                fnp.all(
                    lax.convert_element_type(final_comm_key, aff)
                    == lax.convert_element_type(g_final, aff)
                )
            )
        )

    @parameterized.named_parameters(*_CURVES)
    def test_wrong_commitment_rejected(self, sf: type, curve: curves.Curve) -> None:
        # The Fiat-Shamir now binds the commitment: verifying against a different
        # one rejects (statement binding the bare fold lacked).
        verifier, x, commitment, values, proof = self._commit_open(sf, curve)
        bad = fnp.stack([verifier.key.u])  # U as a stand-in P ≠ the real commitment
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
        coeffs = fnp.array([3, 1, 4, 1], dtype=sf)  # p(x) = 3 + x + 4x² + x³
        x = fnp.array(7, dtype=sf)
        hiding = fnp.array([2, 9, 1, 8], dtype=sf)  # blinding polynomial
        hiding_rand = fnp.array(5, dtype=sf)
        commitment_randomness = fnp.array(11, dtype=sf)
        # The zk path opens a *hiding* commitment ⟨coeffs,G⟩ + cr·s (the open
        # removes all randomness inside the fold).
        commitment, _ = IpaProver(key).commit_zk([coeffs], [commitment_randomness])
        fs = TranscriptChallenger(_transcript(sf), sf)
        _, value, proof, _final_comm_key, _mod_commitment = _open_one_zk(
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

    @parameterized.named_parameters(*_ZK_CURVES)
    def test_zk_value_is_evaluation(self, sf: type, curve: curves.Curve) -> None:
        # Blinding preserves the value: p(7) = 549 despite the random blinding poly.
        _, _, _, value, _ = self._commit_open_zk(sf, curve)
        self.assertEqual(int(value), 549)

    @parameterized.named_parameters(*_ZK_CURVES)
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

    @parameterized.named_parameters(*_ZK_CURVES)
    def test_zk_wrong_value_rejected(self, sf: type, curve: curves.Curve) -> None:
        key, x, commitment, value, proof = self._commit_open_zk(sf, curve)
        bad = value + fnp.array(1, dtype=sf)
        _, claim = reduce_opening_zk(
            key,
            commitment[0],
            x,
            bad,
            proof,
            TranscriptChallenger(_transcript(sf), sf),
        )
        self.assertFalse(bool(settle(key, claim)))

    @parameterized.named_parameters(*_ZK_CURVES)
    def test_zk_open_returns_final_comm_key_and_mod_commitment(
        self, sf: type, curve: curves.Curve
    ) -> None:
        # #371 (zk): _open_one_zk returns the fold's final_comm_key (g[0]) plus the
        # blinded commitment mod_commitment it actually opened. final_comm_key must
        # equal settle's G_final; mod_commitment must equal the blinded commitment
        # (commitment + hc·hiding_comm − s·rand) the verifier independently
        # reconstructs in reduce_opening_zk.
        key = basis.toy_key(curve, n=4)
        coeffs = fnp.array([3, 1, 4, 1], dtype=sf)
        x = fnp.array(7, dtype=sf)
        hiding = fnp.array([2, 9, 1, 8], dtype=sf)
        hiding_rand = fnp.array(5, dtype=sf)
        commitment_randomness = fnp.array(11, dtype=sf)
        commitment, _ = IpaProver(key).commit_zk([coeffs], [commitment_randomness])
        _, value, proof, final_comm_key, mod_commitment = _open_one_zk(
            key,
            commitment[0],
            coeffs,
            x,
            hiding,
            hiding_rand,
            commitment_randomness,
            TranscriptChallenger(_transcript(sf), sf),
        )
        aff = key.basis.dtype

        # final_comm_key == settle's G_final for the zk reduced claim.
        _, claim = reduce_opening_zk(
            key,
            commitment[0],
            x,
            value,
            proof,
            TranscriptChallenger(_transcript(sf), sf),
        )
        s = challenge_vector(claim.u)
        g_final = lax.msm(s, key.basis[: s.shape[0]])
        self.assertTrue(
            bool(
                fnp.all(
                    lax.convert_element_type(final_comm_key, aff)
                    == lax.convert_element_type(g_final, aff)
                )
            )
        )

        # mod_commitment == the blinded commitment the verifier reconstructs from the
        # re-squeezed hiding challenge hc (reduce_opening_zk's own formula).
        one = fnp.ones((), dtype=sf)
        _, hc = TranscriptChallenger(_transcript(sf), sf).hiding_challenge(
            commitment[0], proof.hiding_comm, x, value
        )
        expected_mod = lax.msm(
            fnp.stack([one, hc, -proof.rand]),
            fnp.stack([commitment[0], proof.hiding_comm, key.s]),
        )
        self.assertTrue(
            bool(
                fnp.all(
                    lax.convert_element_type(mod_commitment, aff)
                    == lax.convert_element_type(expected_mod, aff)
                )
            )
        )


class IpaBatchValidationTest(absltest.TestCase):
    # The batch-length guards fire before any MSM, so these need no GPU.
    def test_open_rejects_batch_mismatch(self) -> None:
        sf = zk_dtypes.bn254_sf_mont
        key = basis.toy_key(curves.BN254, n=4)
        coeffs = fnp.array([3, 1, 4, 1], dtype=sf)
        x = fnp.array(7, dtype=sf)
        # commitments is never read — open raises on the length check first.
        dummy = fnp.zeros((1,), dtype=curves.BN254.g1)
        with self.assertRaises(ValueError):
            IpaProver(key).open(
                IpaProverData((coeffs,), dummy), [x, x], _transcript(sf)
            )


if __name__ == "__main__":
    absltest.main()
