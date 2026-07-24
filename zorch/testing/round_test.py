# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round contracts and direct recurrence drivers."""

from __future__ import annotations

import weakref
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.round import (
    Round,
    prove_rounds,
    verify_rounds,
)
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


class _ScaleProver(Round):
    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def __call__(self, carry: Any, transcript: Transcript, _incoming: None) -> Any:
        transcript, sampled = transcript.sample(1)
        return carry * self.factor + sampled[0], transcript, carry


class _ScaleVerifier(Round):
    def __init__(self, factor: int) -> None:
        self.factor = fnp.array(factor, KB)

    def __call__(
        self, carry: Any, transcript: Transcript, msg: Any
    ) -> tuple[Any, Transcript, Any]:
        ok = msg == carry
        transcript, sampled = transcript.sample(1)
        return carry * self.factor + sampled[0], transcript, ok


class _Payload:
    pass


class _ReleaseProbeProver(Round):
    def __init__(
        self,
        payload: _Payload,
        refs: list[weakref.ref[_Payload]],
        live_log: list[int],
    ) -> None:
        self.payload = payload
        self.refs = refs
        self.live_log = live_log

    def __call__(self, carry: Any, transcript: Transcript, _incoming: None) -> Any:
        self.live_log.append(sum(ref() is not None for ref in self.refs))
        transcript, sampled = transcript.sample(1)
        return carry + sampled[0], transcript, carry


class RoundDriverTest(absltest.TestCase):
    def test_roundtrip_heterogeneous(self) -> None:
        factors = (2, 3, 7)
        initial = fnp.array(11, KB)
        final, _, msgs = prove_rounds(
            [_ScaleProver(factor) for factor in factors],
            initial,
            cheap_transcript(KB),
        )
        verified, _, ok = verify_rounds(
            [_ScaleVerifier(factor) for factor in factors],
            initial,
            msgs,
            cheap_transcript(KB),
        )
        self.assertEqual(len(msgs), len(factors))
        self.assertTrue(bool(ok))
        self.assertTrue(bool(verified == final))

    def test_generator_releases_each_round_after_proving(self) -> None:
        refs: list[weakref.ref[_Payload]] = []
        live_log: list[int] = []

        def rounds() -> Any:
            for _ in range(3):
                payload = _Payload()
                refs.append(weakref.ref(payload))
                yield _ReleaseProbeProver(payload, refs, live_log)

        prove_rounds(rounds(), fnp.array(1, KB), cheap_transcript(KB))
        self.assertEqual(live_log, [1, 1, 1])
        self.assertEqual([ref() for ref in refs], [None, None, None])

    def test_generator_matches_eager_rounds(self) -> None:
        factors = (2, 3, 7)
        initial = fnp.array(11, KB)
        eager = prove_rounds(
            [_ScaleProver(factor) for factor in factors],
            initial,
            cheap_transcript(KB),
        )
        lazy = prove_rounds(
            (_ScaleProver(factor) for factor in factors),
            initial,
            cheap_transcript(KB),
        )
        eager_carry, eager_t, eager_msgs = eager
        lazy_carry, lazy_t, lazy_msgs = lazy
        for eager_msg, lazy_msg in zip(eager_msgs, lazy_msgs, strict=True):
            self.assertTrue(bool(eager_msg == lazy_msg))
        self.assertTrue(bool(lazy_carry == eager_carry))
        _, eager_r = eager_t.sample(1)
        _, lazy_r = lazy_t.sample(1)
        self.assertTrue(bool(eager_r[0] == lazy_r[0]))

    def test_verify_rejects_message_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            verify_rounds(
                [_ScaleVerifier(2), _ScaleVerifier(3)],
                fnp.array(1, KB),
                [fnp.array(1, KB)],
                cheap_transcript(KB),
            )

    def test_verify_rejects_tampered_message(self) -> None:
        factors = (2, 3)
        initial = fnp.array(9, KB)
        _, _, msgs = prove_rounds(
            [_ScaleProver(factor) for factor in factors],
            initial,
            cheap_transcript(KB),
        )
        msgs[0] = msgs[0] + fnp.array(1, KB)
        _, _, ok = verify_rounds(
            [_ScaleVerifier(factor) for factor in factors],
            initial,
            msgs,
            cheap_transcript(KB),
        )
        self.assertFalse(bool(ok))


class _FourTupleRound:
    def __call__(
        self, carry: Any, transcript: Transcript, msg: Any
    ) -> tuple[Any, Any, Any, Any]:
        return carry, transcript, carry, carry


class RoundProtocolContractTest(absltest.TestCase):
    def test_verify_driver_rejects_a_four_tuple_result(self) -> None:
        with self.assertRaises(ValueError):
            verify_rounds(
                [_FourTupleRound()],  # type: ignore[list-item]
                fnp.array(1, KB),
                [fnp.array(1, KB)],
                cheap_transcript(KB),
            )

    def test_both_roles_specialize_round(self) -> None:
        prover: Round[Any, None, Any] = _ScaleProver(2)
        verifier: Round[Any, Any, Any] = _ScaleVerifier(2)
        self.assertIsNotNone(prover)
        self.assertIsNotNone(verifier)


if __name__ == "__main__":
    absltest.main()
