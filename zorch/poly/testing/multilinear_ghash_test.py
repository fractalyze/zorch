# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`eval_mle` over `binary_field_ghash` on the CUDA backend (zorch#400).

The eq inner product `eval_mle` sums the hypercube under field addition; a native
reduce-add over a binary field hard-SIGSEGVs on cuda, so it folds a tree of
elementwise adds instead (`zorch.utils.field.field_sum`). This pins that the
result is correct, not merely non-crashing. Sibling of the `mle_coeffs_to_evals`
coverage — that transform already lowered on GPU; the `eval_mle` fold did not.

GPU only: `binary_field_ghash` arithmetic is unlowered on the ZKX CPU PJRT path
(a canonical draw casts to zero there), so a CPU run would be vacuous.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # binary_field_ghash is uint64-backed

import jax.numpy as jnp  # noqa: E402
import zk_dtypes  # noqa: E402
from absl.testing import absltest  # noqa: E402
from jax import Array  # noqa: E402

from zorch.poly.multilinear import eval_mle  # noqa: E402

GH = zk_dtypes.binary_field_ghash  # binary field GF(2^128)

_on_gpu = absltest.skipUnless(
    jax.default_backend() == "gpu",
    "eval_mle over binary_field_ghash exercises the cuda reduce-add lowering the "
    "fold routes around (zorch#400); ghash arithmetic is unlowered on this CPU.",
)


@_on_gpu
class EvalMleGhashTest(absltest.TestCase):
    def test_at_hypercube_vertex_returns_that_eval(self) -> None:
        # eq(vertex_i, vertex_j) = δ_ij, so the eq-weighted sum selects evals[nat].
        # With distinct nonzero evals this fails loudly if the multiply or the
        # tree-sum is wrong — it is not a zeros-equal-zeros pass.
        evals = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=GH)

        def vertex(nat: int) -> Array:  # 3 coords, MSB first, as field 0/1
            return jnp.array([(nat >> (2 - b)) & 1 for b in range(3)], dtype=GH)

        for nat in range(8):
            self.assertTrue(bool(eval_mle(evals, vertex(nat)) == evals[nat]))

    def test_ones_mle_evaluates_to_one(self) -> None:
        # Σ_w eq(w, point) = 1, so an all-ones MLE is 1 at any point — isolates the
        # reduction (every summand is the field one).
        one = jnp.ones((), GH)
        point = jnp.array([5, 6, 7], dtype=GH)
        self.assertTrue(bool(eval_mle(jnp.broadcast_to(one, (8,)), point) == one))

    def test_contracts_nonzero_axis(self) -> None:
        mle = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=GH)  # (2, 4)
        point = jnp.array([5, 6], dtype=GH)
        out = eval_mle(mle, point, axis=1)
        self.assertEqual(out.shape, (2,))
        for r in range(2):
            self.assertTrue(bool(out[r] == eval_mle(mle[r], point)))


if __name__ == "__main__":
    absltest.main()
