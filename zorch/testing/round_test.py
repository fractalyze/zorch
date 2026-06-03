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
from zorch.transcript import StubTranscript, Transcript

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
        if not isinstance(vt, StubTranscript):
            raise AssertionError("expected StubTranscript")
        self.assertEqual(vt.pos, 3)  # one challenge per round

    def test_chain_is_a_round_so_chains_nest(self) -> None:
        ch = jnp.array([1, 2, 3], KB)
        inner = ProveChain([_ScaleProver(2), _ScaleProver(3)])
        outer = ProveChain([inner, _ScaleProver(5)])
        final, t, msgs = outer(jnp.array(4, KB), StubTranscript(ch))
        self.assertEqual(len(msgs), 2)  # [inner's message list, scale-5's message]
        self.assertEqual(len(msgs[0]), 2)  # inner ran two rounds
        if not isinstance(t, StubTranscript):
            raise AssertionError("expected StubTranscript")
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


class _FourTupleRound(Round):
    """Verifier-arity round that returns a 4-tuple, like the per-variable sumcheck
    verifier (which also yields the sampled challenge). Not a `ChainVerifierRound`,
    which replays a 3-tuple -- the shape `VerifyChain` must reject."""

    def __call__(
        self, carry: Any, msg: Any, transcript: Transcript
    ) -> tuple[Any, Any, Any, Any]:
        return carry, transcript, carry, carry


class RoundProtocolContractTest(absltest.TestCase):
    """`ProveChain` / `VerifyChain` accept the typed `ProverRound` /
    `ChainVerifierRound` Protocols, so a structurally wrong round is a *mypy* error,
    not a runtime surprise. These are compile-time assertions: `warn_unused_ignores`
    (pyproject) makes each `# type: ignore` mean "this line MUST be a type error", so
    if a Protocol ever stops biting, mypy fails. The chains are only built, never run.
    The positive direction is covered by `ChainTest` above (it type-checks clean)."""

    def test_prove_chain_rejects_a_verifier_round(self) -> None:
        # A verifier round takes an extra `msg` arg, so it is not a ProverRound.
        chain = ProveChain([_ScaleVerifier(2)])  # type: ignore[list-item]
        self.assertIsInstance(chain, ProveChain)

    def test_verify_chain_rejects_a_prover_round(self) -> None:
        # A prover round lacks the `msg` arg, so it is not a ChainVerifierRound.
        chain = VerifyChain([_ScaleProver(2)])  # type: ignore[list-item]
        self.assertIsInstance(chain, VerifyChain)

    def test_verify_chain_rejects_a_four_tuple_round(self) -> None:
        # The #103 headline: a 4-tuple return is not the 3-tuple VerifyChain replays.
        chain = VerifyChain([_FourTupleRound()])  # type: ignore[list-item]
        self.assertIsInstance(chain, VerifyChain)


if __name__ == "__main__":
    absltest.main()
