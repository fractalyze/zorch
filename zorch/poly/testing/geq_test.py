# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx
import frx.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from frx import Array, lax

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


class VirtualGeqDynamicThresholdTest(absltest.TestCase):
    """The threshold must thread through `jit` / `lax.scan` as a traced leaf:
    the jagged zerocheck round engine carries a `VirtualGeq` whose threshold
    halves each round, so the indicator is a `scan` carry, not a static config."""

    def test_eval_at_traced_threshold_matches_static(self) -> None:
        one = jnp.ones((), KB)
        zero = jnp.zeros((), KB)
        size = 8

        def materialize(threshold: Array) -> Array:
            vg = VirtualGeq(threshold, one, zero)
            return frx.vmap(vg.eval_at)(jnp.arange(size, dtype=jnp.int32))

        for threshold in (0, 3, 7, 8):
            got = frx.jit(materialize)(jnp.int32(threshold))
            want = _materialize(VirtualGeq(threshold, one, zero), size)
            self.assertTrue(bool(jnp.all(got == want)), msg=f"threshold={threshold}")

    def test_scan_fold_traced_threshold_matches_static(self) -> None:
        # The Phase-2 usage: fold the indicator once per round under `lax.scan`
        # with a traced threshold, evaluating at a per-round index. Must match
        # the eager static-int Python fold byte-for-byte.
        num_vars = 4
        one = jnp.ones((), KB)
        zero = jnp.zeros((), KB)
        for threshold in (0, 1, 5, 6, 15, 16):
            alphas = jnp.stack(
                [rand_field(31 * threshold + r, (), KB) for r in range(num_vars)]
            )
            # Per round, hit below / at / above the (halving) threshold.
            idxs = jnp.array(
                [max((threshold >> r) - 1, 0) for r in range(num_vars)], jnp.int32
            )

            def run(thr0: Array, alphas: Array, idxs: Array) -> Array:
                def step(
                    vg: VirtualGeq, xs: tuple[Array, Array]
                ) -> tuple[VirtualGeq, Array]:
                    alpha, idx = xs
                    return vg.fix_last_variable(alpha), vg.eval_at(idx)

                _, outs = lax.scan(step, VirtualGeq(thr0, one, zero), (alphas, idxs))
                return outs

            got = frx.jit(run)(jnp.int32(threshold), alphas, idxs)

            vg = VirtualGeq(threshold, one, zero)
            want = []
            for r in range(num_vars):
                want.append(vg.eval_at(int(idxs[r])))
                vg = vg.fix_last_variable(alphas[r])
            want = jnp.stack(want)
            self.assertTrue(bool(jnp.all(got == want)), msg=f"threshold={threshold}")


class VirtualGeqPytreeTest(absltest.TestCase):
    def test_flatten_unflatten_round_trip(self) -> None:
        vg = VirtualGeq(jnp.int32(3), jnp.ones((), KB), jnp.zeros((), KB))
        leaves, treedef = frx.tree_util.tree_flatten(vg)
        # threshold + geq_coefficient + eq_coefficient — a field misclassified
        # as meta vs data would change this count.
        self.assertLen(leaves, 3)
        back = frx.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(int(back.threshold), 3)

    def test_threads_through_jit_as_argument(self) -> None:
        vg = VirtualGeq(jnp.int32(5), jnp.ones((), KB), jnp.zeros((), KB))
        out = frx.jit(lambda g: g.eval_at(jnp.int32(5)))(vg)
        self.assertEqual(out, vg.eval_at(5))


if __name__ == "__main__":
    absltest.main()
