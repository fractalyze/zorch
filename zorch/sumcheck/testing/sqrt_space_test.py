# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.sumcheck.domain import fold_stacked, product_round_poly
from zorch.sumcheck.sqrt_space import prove_sqrt_space
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

KB = zk_dtypes.koalabear_mont


def _stacked(d: int, l: int) -> jnp.ndarray:
    return jnp.arange(1, d * (1 << l) + 1, dtype=KB).reshape(d, 1 << l)


def _prove_product(p: jnp.ndarray, transcript: Transcript) -> list[jnp.ndarray]:
    """Reference linear-time product sumcheck: fold over the product round message.
    SqrtSpace must reproduce these messages exactly."""
    msgs = []
    for _ in range(log2_strict_usize(p.shape[1])):
        msg = product_round_poly(p)
        transcript, r = transcript.observe_and_sample(msg, 1)
        p = fold_stacked(p, r[0])
        msgs.append(msg)
    return msgs


class SqrtSpaceTest(absltest.TestCase):
    def test_matches_linear_time_prover(self) -> None:
        # The √-space prover must send the exact round polynomials the linear-time
        # product prover does; identical messages keep both transcripts in lockstep,
        # so an independent run from the same seed suffices to compare.
        for d, l in [(2, 4), (3, 4), (2, 5), (2, 6)]:
            p = _stacked(d, l)
            ref = _prove_product(p, cheap_transcript(KB))
            _, _, got = prove_sqrt_space(p, cheap_transcript(KB))
            self.assertLen(got, l)
            for i, (a, b) in enumerate(zip(ref, got, strict=True)):
                self.assertTrue(
                    bool(jnp.array_equal(a, b)), msg=f"d={d} l={l} round {i}"
                )

    def test_prove_folds_to_scalar(self) -> None:
        p_final, _, msgs = prove_sqrt_space(_stacked(3, 4), cheap_transcript(KB))
        self.assertEqual(p_final.shape, (3, 1))
        self.assertLen(msgs, 4)


if __name__ == "__main__":
    absltest.main()
