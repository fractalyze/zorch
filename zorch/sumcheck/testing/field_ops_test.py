# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`NativeFieldOps` reproduces the native operators it abstracts.

The seam exists so a non-native (binary, uint64-lane) field can drive a sumcheck;
this pins that the *native* instantiation is a faithful identity — `add`/`sub`/
`mul`/`sum`/`domain_point` equal `+`/`-`/`*`/`jnp.sum`/`jnp.array(u, dtype)`, the
arithmetic `sumcheck/prover.py` uses — so the two paths cannot drift.
"""
from __future__ import annotations

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.sumcheck.field_ops import NativeFieldOps
from zorch.testkit.random_field import rand_field


class NativeFieldOpsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.ops = NativeFieldOps(F)
        self.a = rand_field(1, (8,), F)
        self.b = rand_field(2, (8,), F)

    def test_add_sub_mul(self) -> None:
        self.assertTrue(jnp.array_equal(self.ops.add(self.a, self.b), self.a + self.b))
        self.assertTrue(jnp.array_equal(self.ops.sub(self.a, self.b), self.a - self.b))
        self.assertTrue(jnp.array_equal(self.ops.mul(self.a, self.b), self.a * self.b))

    def test_sum(self) -> None:
        x = rand_field(3, (4, 8), F)
        self.assertTrue(jnp.array_equal(self.ops.sum(x, axis=-1), jnp.sum(x, axis=-1)))

    def test_identities_and_domain_point(self) -> None:
        self.assertTrue(jnp.array_equal(self.ops.add(self.a, self.ops.zero), self.a))
        self.assertTrue(jnp.array_equal(self.ops.mul(self.a, self.ops.one), self.a))
        for u in (0, 1, 2, 3):
            self.assertTrue(
                jnp.array_equal(self.ops.domain_point(u, self.a), jnp.array(u, F))
            )
        self.assertTrue(
            jnp.array_equal(self.ops.zeros_like(self.a), jnp.zeros_like(self.a))
        )


if __name__ == "__main__":
    absltest.main()
