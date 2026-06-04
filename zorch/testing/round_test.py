# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round base contract + ProveChain / VerifyChain composition.

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
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear


class RoundBaseTest(absltest.TestCase):
    def test_call_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            Round()(None, None)


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
        factors = (2, 3, 7)
        carry0 = jnp.array(11, KB)

        final, _, msgs = ProveChain([_ScaleProver(f) for f in factors])(
            carry0, cheap_transcript(KB)
        )
        self.assertEqual(len(msgs), 3)

        # Prover and verifier each drive a fresh, identical transcript, so they
        # sample the same challenge stream in lockstep — agreeing carries (and ok)
        # are that lockstep, the dual of the old preset-challenge / pos check.
        vcarry, _, ok = VerifyChain([_ScaleVerifier(f) for f in factors])(
            carry0, msgs, cheap_transcript(KB)
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(vcarry == final))  # lockstep carries agree

    def test_chain_is_a_round_so_chains_nest(self) -> None:
        inner = ProveChain([_ScaleProver(2), _ScaleProver(3)])
        outer = ProveChain([inner, _ScaleProver(5)])
        _, _, msgs = outer(jnp.array(4, KB), cheap_transcript(KB))
        self.assertEqual(len(msgs), 2)  # [inner's message list, scale-5's message]
        self.assertEqual(len(msgs[0]), 2)  # inner ran two rounds

    def test_verify_rejects_message_count_mismatch(self) -> None:
        # A short msgs list must fail loud, not silently skip rounds with ok=True.
        chain = VerifyChain([_ScaleVerifier(2), _ScaleVerifier(3)])
        with self.assertRaises(ValueError):
            chain(jnp.array(1, KB), [jnp.array(1, KB)], cheap_transcript(KB))

    def test_verify_rejects_tampered_message(self) -> None:
        factors = (2, 3)
        carry0 = jnp.array(9, KB)
        _, _, msgs = ProveChain([_ScaleProver(f) for f in factors])(
            carry0, cheap_transcript(KB)
        )
        msgs[0] = msgs[0] + jnp.array(1, KB)
        _, _, ok = VerifyChain([_ScaleVerifier(f) for f in factors])(
            carry0, msgs, cheap_transcript(KB)
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
