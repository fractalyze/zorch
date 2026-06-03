# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from typing import Any

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.prove import fold_rounds, prove
from zorch.round import Round
from zorch.sumcheck import prover
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


class _CollectRound(Round):
    """Halves a 1-element-per-factor carry; emits a heterogeneous dict message."""

    def __call__(self, state: Any, transcript: Any) -> Any:
        (xs,) = state
        half = xs.shape[-1] // 2
        msg = {"first": xs[0], "len": xs.shape[-1]}  # non-stackable on purpose
        return [xs[:half]], transcript, msg


class FoldRoundsTest(absltest.TestCase):
    def test_collects_structured_messages_as_list(self) -> None:
        xs = jnp.arange(8, dtype=KB)
        _, _, msgs = fold_rounds(
            _CollectRound(), [xs], StubTranscript(jnp.zeros(0, KB)), 3
        )
        self.assertEqual([m["len"] for m in msgs], [8, 4, 2])
        self.assertEqual(len(msgs), 3)

    def test_prove_rejects_zero_round_state(self) -> None:
        # A width-1 carry derives 0 rounds: the scan would yield no round polys.
        # Fail fast with a clear message instead.
        with self.assertRaisesRegex(ValueError, "at least one round"):
            prove(
                prover.SumcheckRound(1),
                [jnp.arange(1, dtype=KB)],
                StubTranscript(jnp.zeros(0, KB)),
            )

    def test_prove_still_stacks_sumcheck_messages(self) -> None:
        f = jnp.arange(1, 17, dtype=KB)
        ch = jnp.arange(2, 6, dtype=KB)
        _, _, msgs = prove(prover.SumcheckRound(degree=1), [f], StubTranscript(ch))
        self.assertEqual(msgs.round_poly.shape, (4, 2))  # n rounds × (degree+1)
        self.assertEqual(msgs.challenge.shape, (4,))  # one challenge per round


if __name__ == "__main__":
    absltest.main()
