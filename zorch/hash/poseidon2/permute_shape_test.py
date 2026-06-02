"""Poseidon2 implements Permutation and preserves shape/dtype."""

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.hash.permutation import Permutation
from zorch.hash.poseidon2 import Poseidon2, Poseidon2Params


def _params():
    w, er, ir = 16, 4, 20
    return Poseidon2Params(
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


def test_is_a_permutation():
    p = Poseidon2(_params())
    assert isinstance(p, Permutation)
    assert p.width == 16 and p.dtype == F


def test_permute_shape_and_vmap():
    p = Poseidon2(_params())
    x = jnp.arange(16, dtype=F)
    out = p.permute(x)
    assert out.shape == (16,) and out.dtype == F
    batch = jnp.stack([x, x + F(1)])
    bout = jax.vmap(p.permute)(batch)  # thread-per-hash
    assert bout.shape == (2, 16)
    assert jnp.array_equal(bout[0], out)


def test_custom_external_matrix_override_is_rejected():
    base = _params()
    bad = base.external_matrix.at[0, 0].add(F(1))  # non-canonical
    over = Poseidon2Params(
        width=16,
        dtype=F,
        alpha=3,
        external_rounds=4,
        internal_rounds=20,
        external_constants_initial=base.external_constants_initial,
        external_constants_terminal=base.external_constants_terminal,
        internal_constants=base.internal_constants,
        internal_diag=base.internal_diag,
        external_matrix=bad,
    )
    try:
        Poseidon2(over)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError on custom external_matrix")


if __name__ == "__main__":
    test_is_a_permutation()
    test_permute_shape_and_vmap()
    test_custom_external_matrix_override_is_rejected()
    print("ok")
