# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.poly.geq import VirtualGeq
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear_mont


def _materialize(vg: VirtualGeq, size: int) -> Array:
    return jnp.stack([vg.eval_at(i) for i in range(size)])


class VirtualGeqTest(absltest.TestCase):
    def test_materializes_indicator(self) -> None:
        one = jnp.ones((), KB)
        zero = jnp.zeros((), KB)
        vg = VirtualGeq(3, one, zero)
        want = jnp.array([0, 0, 0, 1, 1, 1, 1, 1], KB)
        self.assertTrue(bool(jnp.all(_materialize(vg, 8) == want)))

    def test_fold_matches_materialized_even_odd(self) -> None:
        # Repeated fix_last_variable == the even/odd partial-eval bind of the
        # materialized vector, across even/odd thresholds and both edges.
        num_vars = 4
        one = jnp.ones((), KB)
        zero = jnp.zeros((), KB)
        for threshold in (0, 1, 5, 6, 15, 16):
            vg = VirtualGeq(threshold, one, zero)
            v = _materialize(vg, 1 << num_vars)
            for r in range(num_vars):
                alpha = rand_field(31 * threshold + r, (), KB)
                vg = vg.fix_last_variable(alpha)
                v = v[0::2] + alpha * (v[1::2] - v[0::2])
                self.assertTrue(
                    bool(jnp.all(_materialize(vg, 1 << (num_vars - 1 - r)) == v)),
                    msg=f"threshold={threshold}, round={r}",
                )


if __name__ == "__main__":
    absltest.main()
