"""Poseidon2 implements Permutation and preserves shape/dtype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from zk_dtypes import koalabear_mont as F

from zorch.hash.permutation import Permutation
from zorch.hash.poseidon2 import Poseidon2, Poseidon2Params


def _params() -> Poseidon2Params:
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


def test_is_a_permutation() -> None:
    p = Poseidon2(_params())
    assert isinstance(p, Permutation)
    assert p.width == 16 and p.dtype == F


def test_permute_shape_and_vmap() -> None:
    p = Poseidon2(_params())
    x = jnp.arange(16, dtype=F)
    out = p.permute(x)
    assert out.shape == (16,) and out.dtype == F
    batch = jnp.stack([x, x + F(1)])
    bout = jax.vmap(p.permute)(batch)  # thread-per-hash
    assert bout.shape == (2, 16) and bout.dtype == F
    assert jnp.array_equal(bout[0], out)


def test_custom_external_matrix_is_applied() -> None:
    base = _params()
    assert base.external_matrix is not None
    custom = base.external_matrix.at[0, 0].add(
        F(1)
    )  # a different valid MDS-shaped matrix
    over = Poseidon2(Poseidon2Params(**{**vars(base), "external_matrix": custom}))
    x = jnp.arange(16, dtype=F)
    # external_matrix is an operand (external_matrix @ state), so a different
    # matrix produces a different permutation — the override is genuinely used.
    assert not jnp.array_equal(over.permute(x), Poseidon2(base).permute(x))


def test_permute_rejects_wrong_shape() -> None:
    p = Poseidon2(_params())
    with pytest.raises(ValueError):
        p.permute(jnp.zeros((15,), dtype=F))  # width != 16
    with pytest.raises(ValueError):
        p.permute(jnp.zeros((2, 16), dtype=F))  # batched, not a 1-D state


if __name__ == "__main__":
    test_is_a_permutation()
    test_permute_shape_and_vmap()
    test_custom_external_matrix_is_applied()
    test_permute_rejects_wrong_shape()
    print("ok")
