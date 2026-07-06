# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The TensorCode seam, checked against an independent multilinear oracle.

The oracle shares no code with `eval_point`: it converts the message coefficients
to the hypercube-evaluation basis (`mle_coeffs_to_evals`) and evaluates that MLE
(`eval_mle`) at each returned point, then asserts equality with the encoder's
codeword. That pins the tensor-code fact `encode(w)[s] == ŵ(p_s)` Ligerito
recurses on, independent of the geometric-tensor construction inside
`eval_point`. The coset case checks the map tracks the
shifted evaluation domain, not just the base subgroup.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized

from zorch.coding.reed_solomon import ReedSolomon
from zorch.coding.tensor_code import TensorCode
from zorch.poly.multilinear import eval_mle, mle_coeffs_to_evals
from zorch.testkit.random_field import rand_field

F = zk_dtypes.koalabear_mont


class TensorCodeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="k2", message_len=4, blowup=2, coset=False),
        dict(testcase_name="k3", message_len=8, blowup=2, coset=False),
        dict(testcase_name="k3_rate4", message_len=8, blowup=4, coset=False),
        dict(testcase_name="k4", message_len=16, blowup=2, coset=False),
        dict(testcase_name="k3_coset", message_len=8, blowup=2, coset=True),
    )
    def test_eval_point_matches_mle(
        self, message_len: int, blowup: int, coset: bool
    ) -> None:
        shift = jnp.asarray(3, F) if coset else None  # arbitrary non-trivial coset
        code = ReedSolomon(message_len, blowup, F, coset_shift=shift)
        self.assertIsInstance(code, TensorCode)

        w = rand_field(0, (message_len,), F)
        codeword = code.encode(w)  # (block_len,)
        positions = jnp.arange(code.block_len)
        points = code.eval_point(positions)  # (block_len, k)

        # Independent oracle: message-coeff -> hypercube-eval basis, then eval.
        evals = mle_coeffs_to_evals(w)
        oracle = jax.vmap(lambda p: eval_mle(evals, p))(points)  # (block_len,)
        self.assertTrue(bool(jnp.all(oracle == codeword)))

    def test_eval_point_shape(self) -> None:
        code = ReedSolomon(8, 2, F)  # k = 3
        points = code.eval_point(jnp.array([0, 1, 5]))
        self.assertEqual(points.shape, (3, 3))


if __name__ == "__main__":
    absltest.main()
