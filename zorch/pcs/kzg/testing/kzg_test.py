# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.pcs.kzg.prover import KzgProver, KzgProverData, _quotient_and_eval
from zorch.pcs.kzg.testing.srs import toy_srs
from zorch.pcs.kzg.verifier import KzgVerifier
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

# The quotient/eval math is field-agnostic; exercise it over a small base field
# (CPU-friendly), independent of bn254 and the GPU msm/pairing path.
KB = zk_dtypes.koalabear_mont


def _horner(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """f(x) for ascending `coeffs`, by index loop (iterating a field array under
    CUDA dispatches lax.sign — avoid it even though this test is CPU)."""
    acc = jnp.zeros((), coeffs.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * x + coeffs[i]
    return acc


class QuotientAndEvalTest(absltest.TestCase):
    def test_remainder_is_evaluation(self) -> None:
        # f(x) = 3 + 5x + 2x² + 7x³; the synthetic-division remainder is f(z).
        coeffs = jnp.array([3, 5, 2, 7], dtype=KB)
        z = jnp.array(4, dtype=KB)
        q, fz = _quotient_and_eval(coeffs, z)
        self.assertEqual(q.shape, (3,))
        self.assertEqual(int(fz), int(_horner(coeffs, z)))

    def test_division_identity(self) -> None:
        # q(x)·(x − z) + f(z) == f(x) for every x.
        coeffs = jnp.array([3, 5, 2, 7], dtype=KB)
        z = jnp.array(4, dtype=KB)
        q, fz = _quotient_and_eval(coeffs, z)
        for xv in (0, 1, 6, 100):
            x = jnp.array(xv, dtype=KB)
            lhs = _horner(q, x) * (x - z) + fz
            self.assertEqual(int(lhs), int(_horner(coeffs, x)))

    def test_degree_one(self) -> None:
        # f(x) = a0 + a1 x → q = a1 (constant), f(z) = a0 + a1 z.
        coeffs = jnp.array([9, 2], dtype=KB)
        z = jnp.array(5, dtype=KB)
        q, fz = _quotient_and_eval(coeffs, z)
        self.assertEqual(q.shape, (1,))
        self.assertEqual(int(q[0]), 2)
        self.assertEqual(int(fz), int(jnp.array(9, KB) + jnp.array(2, KB) * z))

    def test_constant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _quotient_and_eval(jnp.array([7], dtype=KB), jnp.array(1, dtype=KB))


# --- full commit -> open -> verify over bn254 (GPU: lax.msm is GPU-only) ---------

# Standard domain until lax.pairing_check decodes Montgomery inputs (zkx#518);
# see the note in testing/srs.py.
SF = zk_dtypes.bn254_sf
G1 = zk_dtypes.bn254_g1_affine

_GPU = jax.default_backend() == "gpu"


def _transcript() -> DuplexTranscript:
    return cheap_transcript(SF)  # KZG threads a transcript but doesn't use it


@absltest.skipUnless(_GPU, "KZG commit/open use lax.msm, a GPU-only kernel")
class KzgRoundTripTest(absltest.TestCase):
    def setUp(self) -> None:
        self.pk, self.vk = toy_srs(tau=5, n=4)
        self.coeffs = jnp.array([3, 1, 4, 1], dtype=SF)  # f(x) = 3 + x + 4x² + x³
        self.z = jnp.array(7, dtype=SF)
        self.prover = KzgProver(self.pk)
        self.verifier = KzgVerifier(self.vk)

    def _open(self) -> tuple[Array, Array, Array]:
        commitment, data = self.prover.commit([self.coeffs])
        values, proof, _ = self.prover.open(data, [self.z], _transcript())
        return commitment, values, proof

    def test_open_verifies(self) -> None:
        commitment, values, proof = self._open()
        ok, _ = self.verifier.verify(commitment, [self.z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_wrong_value_rejected(self) -> None:
        commitment, values, proof = self._open()
        bad = values + jnp.array(1, dtype=SF)
        ok, _ = self.verifier.verify(commitment, [self.z], bad, proof, _transcript())
        self.assertFalse(bool(ok))


class KzgBatchValidationTest(absltest.TestCase):
    # The batch-length guard fires before any MSM, so these need no GPU.
    def test_open_rejects_batch_mismatch(self) -> None:
        pk, _ = toy_srs(tau=5, n=4)
        coeffs = jnp.array([3, 1, 4, 1], dtype=SF)
        z = jnp.array(7, dtype=SF)
        with self.assertRaises(ValueError):
            KzgProver(pk).open(KzgProverData((coeffs,)), [z, z], _transcript())

    def test_verify_rejects_batch_mismatch(self) -> None:
        _, vk = toy_srs(tau=5, n=4)
        z = jnp.array(7, dtype=SF)
        one_commit = jnp.stack([jnp.asarray(G1((1, 2)))])
        with self.assertRaises(ValueError):
            KzgVerifier(vk).verify(
                one_commit,
                [z, z],
                jnp.zeros(2, SF),
                jnp.zeros(2, SF),
                _transcript(),
            )


if __name__ == "__main__":
    absltest.main()
