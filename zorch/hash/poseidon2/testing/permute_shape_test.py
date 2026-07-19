"""Poseidon2 implements Permutation and preserves shape/dtype."""

from __future__ import annotations

import dataclasses

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.permutation import Permutation
from zorch.hash.poseidon2.params import Poseidon2Params
from zorch.hash.poseidon2.poseidon2 import Poseidon2


def _params() -> Poseidon2Params:
    w, er, ir = 16, 4, 20
    return Poseidon2Params(
        width=w,
        dtype=F,
        alpha=3,
        external_rounds=er,
        internal_rounds=ir,
        external_constants_initial=fnp.zeros((er, w), dtype=F),
        external_constants_terminal=fnp.zeros((er, w), dtype=F),
        internal_constants=fnp.zeros((ir, w), dtype=F),
        internal_diag=fnp.ones((w,), dtype=F),
    )


class Poseidon2InternalJScaleTest(absltest.TestCase):
    """`internal_j_scale` generalizes the internal matrix to
    c*J + Diag(internal_diag); the default must stay byte-identical to the
    historical J + Diag form."""

    def test_explicit_one_equals_default(self) -> None:
        base = _params()
        scaled = dataclasses.replace(base, internal_j_scale=fnp.ones((), dtype=F))
        x = fnp.arange(16, dtype=fnp.uint32).view(F)
        self.assertTrue(
            bool(fnp.all(Poseidon2(base).permute(x) == Poseidon2(scaled).permute(x)))
        )

    def test_non_unit_scale_changes_output(self) -> None:
        base = _params()
        scaled = dataclasses.replace(base, internal_j_scale=fnp.full((), 2, dtype=F))
        x = fnp.arange(16, dtype=fnp.uint32).view(F)
        self.assertFalse(
            bool(fnp.all(Poseidon2(base).permute(x) == Poseidon2(scaled).permute(x)))
        )


class Poseidon2PermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        p = Poseidon2(_params())
        self.assertIsInstance(p, Permutation)
        self.assertEqual(p.width, 16)
        self.assertEqual(p.dtype, F)

    def test_permute_shape_and_vmap(self) -> None:
        p = Poseidon2(_params())
        x = fnp.arange(16, dtype=F)
        out = p.permute(x)
        self.assertEqual(out.shape, (16,))
        self.assertEqual(out.dtype, F)
        batch = fnp.stack([x, x + F(1)])
        bout = frx.vmap(p.permute)(batch)  # thread-per-hash
        self.assertEqual(bout.shape, (2, 16))
        self.assertEqual(bout.dtype, F)
        self.assertTrue(bool(fnp.array_equal(bout[0], out)))

    def test_custom_external_matrix_is_applied(self) -> None:
        base = _params()
        ext = base.external_matrix
        if ext is None:
            raise AssertionError("external_matrix should default to canonical")
        custom = ext.at[0, 0].add(F(1))  # a different valid MDS-shaped matrix
        over = Poseidon2(Poseidon2Params(**{**vars(base), "external_matrix": custom}))
        x = fnp.arange(16, dtype=F)
        # external_matrix is an operand (external_matrix @ state), so a different
        # matrix produces a different permutation — the override is genuinely used.
        self.assertFalse(
            bool(fnp.array_equal(over.permute(x), Poseidon2(base).permute(x)))
        )

    def test_permute_rejects_wrong_shape(self) -> None:
        p = Poseidon2(_params())
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((15,), dtype=F))  # width != 16
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((2, 16), dtype=F))  # batched, not a 1-D state


if __name__ == "__main__":
    absltest.main()
