"""Poseidon2Params: fully-free surface + fail-loud validation."""

import jax.numpy as jnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.params import Poseidon2Params


def _good(**over):
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


def test_external_matrix_defaults_to_canonical():
    p = _good()
    assert p.external_matrix.shape == (16, 16)
    assert p.external_matrix.dtype == F


def test_bad_rc_shape_raises():
    try:
        _good(internal_constants=jnp.zeros((19, 16), dtype=F))  # wrong round count
    except ValueError as e:
        assert "internal_constants" in str(e)
    else:
        raise AssertionError("expected ValueError on wrong internal_constants shape")


def test_nonzero_internal_lane_raises():
    bad = np.zeros((20, 16), dtype=np.int32)
    bad[0, 1] = 1  # lane 1 nonzero
    try:
        _good(internal_constants=jnp.array(bad, dtype=F))
    except ValueError as e:
        assert "lane" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on nonzero internal lane 1..w-1")


if __name__ == "__main__":
    test_external_matrix_defaults_to_canonical()
    test_bad_rc_shape_raises()
    test_nonzero_internal_lane_raises()
    print("ok")
