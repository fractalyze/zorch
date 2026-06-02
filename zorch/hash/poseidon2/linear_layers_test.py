"""Normal-form linear layers equal the jnp.dot matrix form, any field/diagonal."""

import jax.numpy as jnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.params import _mds_external_default
from zorch.hash.poseidon2.poseidon2 import _external_linear, _internal_linear

P = 0x7F000001
WIDTHS = (4, 8, 12, 16, 20, 24)


def _rand_vec(w, rng):
    return jnp.array(rng.integers(0, P, size=w, dtype=np.int64), dtype=F)


def test_external_normal_form_equals_matrix():
    rng = np.random.default_rng(0)
    for w in WIDTHS:
        m = _mds_external_default(w, F)
        for _ in range(5):
            x = _rand_vec(w, rng)
            assert jnp.array_equal(_external_linear(x, w), jnp.dot(m, x)), w


def test_internal_normal_form_equals_matrix():
    rng = np.random.default_rng(1)
    one = np.array(F(1))
    for w in WIDTHS:
        for _ in range(5):
            v_canon = rng.integers(0, P, size=w, dtype=np.int64)
            v = jnp.array(v_canon, dtype=F)
            m = np.full((w, w), one)
            for i in range(w):
                m[i, i] = np.array(F(int(v_canon[i]))) + one  # V[i] + 1 on the diagonal
            m = jnp.array(m, dtype=F)
            x = _rand_vec(w, rng)
            assert jnp.array_equal(_internal_linear(x, v, w), jnp.dot(m, x)), w


if __name__ == "__main__":
    test_external_normal_form_equals_matrix()
    test_internal_normal_form_equals_matrix()
    print("ok")
