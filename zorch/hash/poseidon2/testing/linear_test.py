"""Normal-form linear layers equal the matrix form and stay fusion-ready."""

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.linear import (
    apply_external_standard,
    apply_internal,
    apply_matrix,
)
from zorch.hash.poseidon2.params import default_external_matrix
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field


class LinearLayerTest(absltest.TestCase):
    def test_apply_matrix_equals_matmul(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        s = rand_field(2, (w,), F)
        self.assertTrue(bool(jnp.array_equal(apply_matrix(m, s), m @ s)))

    def test_apply_internal_equals_jdiag(self) -> None:
        w = 16
        d = rand_field(3, (w,), F)
        s = rand_field(4, (w,), F)
        m_int = jnp.ones((w, w), dtype=F) + jnp.diag(d)
        self.assertTrue(bool(jnp.array_equal(apply_internal(d, s), m_int @ s)))

    def test_apply_external_standard_equals_default_matrix(self) -> None:
        # The literal-coefficient external layer must byte-match the array form
        # apply_matrix(default_external_matrix(w)) it stands in for.
        for w in (8, 16):
            m = default_external_matrix(w, F)
            s = rand_field(7, (w,), F)
            self.assertTrue(
                bool(jnp.array_equal(apply_external_standard(s), apply_matrix(m, s))),
                f"width {w}",
            )

    def test_apply_external_standard_rejects_bad_width(self) -> None:
        with self.assertRaises(ValueError):
            apply_external_standard(rand_field(2, (6,), F))  # 6 % 4 != 0
        with self.assertRaises(ValueError):
            apply_external_standard(
                rand_field(2, (16, 16), F)
            )  # 2-D, not a lane vector

    def test_apply_matrix_rejects_mismatched_state(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        with self.assertRaises(ValueError):
            apply_matrix(m, rand_field(2, (w, w), F))  # 2-D, not a lane vector
        with self.assertRaises(ValueError):
            apply_matrix(m, rand_field(2, (w + 1,), F))  # length mismatches matrix

    def test_apply_internal_rejects_mismatched_state(self) -> None:
        w = 16
        d = rand_field(3, (w,), F)
        with self.assertRaises(ValueError):
            apply_internal(d, rand_field(4, (w + 1,), F))  # length mismatches diag

    def test_normal_form_is_fusion_ready(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        d = rand_field(3, (w,), F)
        s = rand_field(2, (w,), F)
        # Element-wise only — no reduce/dot/gather boundary (whitelist gate).
        assert_fusion_ready(lambda v: apply_matrix(m, v), s, reduces=0)
        assert_fusion_ready(lambda v: apply_internal(d, v), s, reduces=0)
        assert_fusion_ready(apply_external_standard, s, reduces=0)
        # The matrix form reduces, so the gate must bite — else the check is vacuous.
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda v: m @ v, s, reduces=0)


if __name__ == "__main__":
    absltest.main()
