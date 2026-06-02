"""Normal-form linear layers equal the matrix form and stay fusion-ready."""

import jax.numpy as jnp
import pytest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.linear import apply_internal, apply_matrix
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field


def test_apply_matrix_equals_matmul() -> None:
    w = 16
    m = rand_field(1, (w, w), F)
    s = rand_field(2, (w,), F)
    assert jnp.array_equal(apply_matrix(m, s), m @ s)


def test_apply_internal_equals_jdiag() -> None:
    w = 16
    d = rand_field(3, (w,), F)
    s = rand_field(4, (w,), F)
    m_int = jnp.ones((w, w), dtype=F) + jnp.diag(d)
    assert jnp.array_equal(apply_internal(d, s), m_int @ s)


def test_normal_form_is_fusion_ready() -> None:
    w = 16
    m = rand_field(1, (w, w), F)
    d = rand_field(3, (w,), F)
    s = rand_field(2, (w,), F)
    # Element-wise only — no reduce/dot/gather boundary (whitelist gate).
    assert_fusion_ready(lambda v: apply_matrix(m, v), s, reduces=0)
    assert_fusion_ready(lambda v: apply_internal(d, v), s, reduces=0)
    # The matrix form reduces, so the gate must bite — else the check is vacuous.
    with pytest.raises(AssertionError):
        assert_fusion_ready(lambda v: m @ v, s, reduces=0)


if __name__ == "__main__":
    test_apply_matrix_equals_matmul()
    test_apply_internal_equals_jdiag()
    test_normal_form_is_fusion_ready()
    print("ok")
