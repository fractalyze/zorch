# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""ProveChain / VerifyChain composition, exercised with toy rounds.

The toy round threads a scalar carry and a sampled challenge so the test covers
what the chains must guarantee: carry threading, lockstep transcript threading
(prover and verifier sample in the same order), message collect/consume, ok
aggregation, heterogeneous rounds, and nesting (a chain is itself a Round).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.round import ProveChain, Round, VerifyChain
from zorch.transcript import StubTranscript, Transcript

KB = zk_dtypes.koalabear


class _ScaleProver(Round):
    """carry -> carry*factor + r; emits the input carry as its message."""

    def __init__(self, factor: int) -> None:
        self.factor = jnp.array(factor, KB)

    def __call__(self, carry: Any, transcript: Transcript) -> Any:
        transcript, r = transcript.sample(1)
        return carry * self.factor + r[0], transcript, carry


class _ScaleVerifier(Round):
    """Dual of `_ScaleProver`: replays the update, checks the emitted carry."""

    def __init__(self, factor: int) -> None:
        self.factor = jnp.array(factor, KB)

    def __call__(self, carry: Any, msg: Any, transcript: Transcript) -> Any:
        ok = msg == carry
        transcript, r = transcript.sample(1)
        return carry * self.factor + r[0], transcript, ok


class ChainTest(absltest.TestCase):
    def test_roundtrip_heterogeneous(self) -> None:
        ch = jnp.array([2, 3, 5], KB)
        factors = (2, 3, 7)
        carry0 = jnp.array(11, KB)

        final, _, msgs = ProveChain([_ScaleProver(f) for f in factors])(
            carry0, StubTranscript(ch)
        )
        self.assertEqual(len(msgs), 3)

        vcarry, vt, ok = VerifyChain([_ScaleVerifier(f) for f in factors])(
            carry0, msgs, StubTranscript(ch)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(vcarry == final))  # lockstep carries agree
        assert isinstance(vt, StubTranscript)
        self.assertEqual(vt.pos, 3)  # one challenge per round

    def test_chain_is_a_round_so_chains_nest(self) -> None:
        ch = jnp.array([1, 2, 3], KB)
        inner = ProveChain([_ScaleProver(2), _ScaleProver(3)])
        outer = ProveChain([inner, _ScaleProver(5)])
        final, t, msgs = outer(jnp.array(4, KB), StubTranscript(ch))
        self.assertEqual(len(msgs), 2)  # [inner's message list, scale-5's message]
        self.assertEqual(len(msgs[0]), 2)  # inner ran two rounds
        assert isinstance(t, StubTranscript)
        self.assertEqual(t.pos, 3)  # 2 inner + 1 outer challenges

    def test_verify_rejects_message_count_mismatch(self) -> None:
        # A short msgs list must fail loud, not silently skip rounds with ok=True.
        chain = VerifyChain([_ScaleVerifier(2), _ScaleVerifier(3)])
        with self.assertRaises(ValueError):
            chain(
                jnp.array(1, KB), [jnp.array(1, KB)], StubTranscript(jnp.zeros(2, KB))
            )

    def test_verify_rejects_tampered_message(self) -> None:
        ch = jnp.array([2, 3], KB)
        factors = (2, 3)
        carry0 = jnp.array(9, KB)
        _, _, msgs = ProveChain([_ScaleProver(f) for f in factors])(
            carry0, StubTranscript(ch)
        )
        msgs[0] = msgs[0] + jnp.array(1, KB)
        _, _, ok = VerifyChain([_ScaleVerifier(f) for f in factors])(
            carry0, msgs, StubTranscript(ch)
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
