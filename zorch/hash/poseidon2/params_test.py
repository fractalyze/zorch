"""Poseidon2Params: fully-free surface + fail-loud validation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.params import Poseidon2Params


def _good(**over: Any) -> Poseidon2Params:
    w, er, ir = 16, 4, 20
    base = dict(
        width=w,
        dtype=F,
        alpha=3,
        external_rounds=er,
        internal_rounds=ir,
        external_constants_initial=jnp.zeros((er, w), dtype=F),
        external_constants_terminal=jnp.zeros((er, w), dtype=F),
        internal_constants=jnp.zeros((ir, w), dtype=F),
        internal_diag=jnp.ones((w,), dtype=F),
    )
    base.update(over)
    return Poseidon2Params(**base)


class Poseidon2ParamsTest(absltest.TestCase):
    def test_external_matrix_defaults_to_canonical(self) -> None:
        p = _good()
        ext = p.external_matrix
        if ext is None:
            raise AssertionError("external_matrix should default to canonical")
        self.assertEqual(ext.shape, (16, 16))
        self.assertEqual(ext.dtype, F)

    def test_bad_rc_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(internal_constants=jnp.zeros((19, 16), dtype=F))  # wrong round count
        self.assertIn("internal_constants", str(cm.exception))

    def test_nonzero_internal_lane_raises(self) -> None:
        bad = np.zeros((20, 16), dtype=np.int32)
        bad[0, 1] = 1  # lane 1 nonzero
        with self.assertRaises(ValueError) as cm:
            _good(internal_constants=jnp.array(bad, dtype=F))
        self.assertIn("lane", str(cm.exception).lower())


if __name__ == "__main__":
    absltest.main()
