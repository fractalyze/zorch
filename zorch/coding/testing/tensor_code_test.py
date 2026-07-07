# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The TensorCode seam, checked against an independent multilinear oracle.

The oracle shares no code with `eval_point`: it converts the message coefficients
to the hypercube-evaluation basis (`mle_coeffs_to_evals`) and evaluates that MLE
(`eval_mle`) at each returned point, then asserts equality with the encoder's
codeword. That pins the tensor-code fact `encode(w)[s] == ŵ(p_s)` Ligerito
recurses on, independent of the geometric-tensor construction inside
`eval_point`. The coset case checks the map tracks the shifted evaluation domain.

Reed-Solomon realizes the seam over both a prime field (multiplicative NTT,
geometric tensor) and a binary field (`lax.ntt` runs the LCH additive NTT, so the
tensor is the subspace polynomials `Ŵ_i`) — same code, same `encode`, the
`eval_point` factorization is the only field-dependent piece.
"""
from __future__ import annotations

from typing import Any

import jax

jax.config.update("jax_enable_x64", True)  # binary_field_ghash is uint64-backed

import jax.numpy as jnp  # noqa: E402
import zk_dtypes  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from zorch.coding.reed_solomon import ReedSolomon  # noqa: E402
from zorch.coding.tensor_code import TensorCode  # noqa: E402
from zorch.poly.multilinear import eval_mle, mle_coeffs_to_evals  # noqa: E402
from zorch.testkit.random_field import rand_field  # noqa: E402

KB = zk_dtypes.koalabear_mont  # prime field: multiplicative NTT
GH = jnp.binary_field_ghash  # binary field: additive (LCH) NTT


class TensorCodeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="rs_k2", message_len=4, blowup=2, coset=False, dtype=KB),
        dict(testcase_name="rs_k3", message_len=8, blowup=2, coset=False, dtype=KB),
        dict(
            testcase_name="rs_k3_rate4", message_len=8, blowup=4, coset=False, dtype=KB
        ),
        dict(testcase_name="rs_k4", message_len=16, blowup=2, coset=False, dtype=KB),
        dict(
            testcase_name="rs_k3_coset", message_len=8, blowup=2, coset=True, dtype=KB
        ),
        dict(
            testcase_name="additive_k2", message_len=4, blowup=2, coset=False, dtype=GH
        ),
        dict(
            testcase_name="additive_k3", message_len=8, blowup=2, coset=False, dtype=GH
        ),
        dict(
            testcase_name="additive_k3_rate4",
            message_len=8,
            blowup=4,
            coset=False,
            dtype=GH,
        ),
        dict(
            testcase_name="additive_k4", message_len=16, blowup=2, coset=False, dtype=GH
        ),
    )
    def test_eval_point_matches_mle(
        self, message_len: int, blowup: int, coset: bool, dtype: Any
    ) -> None:
        shift = jnp.asarray(3, dtype) if coset else None  # arbitrary non-trivial coset
        code = ReedSolomon(message_len, blowup, dtype, coset_shift=shift)
        self.assertIsInstance(code, TensorCode)

        w = rand_field(0, (message_len,), dtype)
        codeword = code.encode(w)  # (block_len,)
        positions = jnp.arange(code.block_len)
        points = code.eval_point(positions)  # (block_len, k)

        # Independent oracle: message-coeff -> hypercube-eval basis, then eval.
        evals = mle_coeffs_to_evals(w)
        oracle = jax.vmap(lambda p: eval_mle(evals, p))(points)  # (block_len,)
        self.assertTrue(bool(jnp.all(oracle == codeword)))

    @parameterized.named_parameters(
        dict(testcase_name="prime", dtype=KB),
        dict(testcase_name="binary", dtype=GH),
    )
    def test_eval_point_shape(self, dtype: Any) -> None:
        code = ReedSolomon(8, 2, dtype)  # k = 3
        points = code.eval_point(jnp.array([0, 1, 5]))
        self.assertEqual(points.shape, (3, 3))

    def test_binary_field_fold_unimplemented(self) -> None:
        # A binary-field Reed-Solomon is a TensorCode, not a FoldableCode: the
        # multiplicative FRI fold has no additive analog here, so it must reject
        # rather than fold over the wrong (multiplicative) domain.
        code = ReedSolomon(8, 2, GH)
        codeword = code.encode(rand_field(0, (8,), GH))
        with self.assertRaises(NotImplementedError):
            code.fold(codeword, jnp.asarray(2, GH))

    def test_binary_field_coset_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReedSolomon(8, 2, GH, coset_shift=jnp.asarray(3, GH))

    def test_binary_eval_table_lazy_and_jit_safe(self) -> None:
        # Construction must not build the additive eval-point table — encode-only
        # consumers (Ligero matrix commits) never read it, and per-level code
        # factories rely on cheap construction. A first eval_point under jit must
        # cache a CONCRETE table: a leaked tracer would fail the eager reuse.
        code = ReedSolomon(8, 2, GH)
        self.assertIsNone(code._binary_eval_table)

        positions = jnp.array([0, 1, 5])
        under_jit = jax.jit(code.eval_point)(positions)
        reused = code.eval_point(positions)  # eager, reuses the cached table
        self.assertTrue(bool(jnp.all(under_jit == reused)))


if __name__ == "__main__":
    absltest.main()
