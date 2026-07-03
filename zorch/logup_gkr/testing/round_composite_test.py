# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""zorch#327 (Z1 prototype): the FS-less `zorch.sumcheck.round` marker around the
dense interaction fold+sum is byte-identical to the eager `_fix_and_sum_int`, and
emits the composite with the pinned operand/attr ABI. This is the marker contract
the xla `SumcheckRecognizer` extension (fractalyze/xla#179) must accept."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.jagged_prover import (
    _composite_fix_and_sum_dense,
    _fix_and_sum_int,
    _InterpConsts,
    _Planes,
    _round_interp_constants,
    _RoundScalars,
)
from zorch.sumcheck.prover import SUMCHECK_ROUND_MARKER
from zorch.testkit.random_field import rand_field

KB = zk_dtypes.koalabear_mont


def _inputs(
    m: int = 8,
) -> tuple[_Planes, jax.Array, jax.Array, _RoundScalars, _InterpConsts]:
    """A random even-width dense-interaction round: four MLE planes, the eq table,
    the previous challenge, the round scalars, and the interp constants."""
    planes = _Planes(*(rand_field(s, (m,), KB) for s in range(4)))
    eq_int = rand_field(10, (m,), KB)
    alpha = rand_field(11, (), KB)
    scalars = _RoundScalars(
        eq_adj=rand_field(12, (), KB),
        pad_adj=rand_field(13, (), KB),
        z_cur=rand_field(14, (), KB),
        claim=rand_field(15, (), KB),
        lam=rand_field(16, (), KB),
    )
    consts = _InterpConsts(*_round_interp_constants(KB))
    return planes, eq_int, alpha, scalars, consts


class RoundCompositeTest(absltest.TestCase):
    def test_byte_identical_to_eager(self) -> None:
        planes, eq_int, alpha, scalars, consts = _inputs()
        want = _fix_and_sum_int(planes, eq_int, alpha, scalars, consts)
        got = _composite_fix_and_sum_dense(planes, eq_int, alpha, scalars, consts)
        got_leaves = jax.tree_util.tree_leaves(got)
        want_leaves = jax.tree_util.tree_leaves(want)
        self.assertEqual(len(got_leaves), len(want_leaves))
        for g, w in zip(got_leaves, want_leaves):
            self.assertTrue(bool(jnp.all(g == w)), "marked round diverged from eager")

    def test_emits_marker_with_abi(self) -> None:
        planes, eq_int, alpha, scalars, consts = _inputs()
        # `_InterpConsts` is not a pytree (dtype-derived constants), so close over
        # it and trace only the array operands.
        jaxpr = jax.make_jaxpr(
            lambda p, e, a, s: _composite_fix_and_sum_dense(p, e, a, s, consts)
        )(planes, eq_int, alpha, scalars)
        text = jaxpr.pretty_print()
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # The phase/variant attributes are the recognizer's routing key.
        self.assertIn("mid", text)
        self.assertIn("dense", text)


if __name__ == "__main__":
    absltest.main()
