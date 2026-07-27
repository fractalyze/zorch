# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-variable verifier compiles to one traced region, flat in trip count.

Issue #58: `verify` must lower its round loop to a single `lax.scan`, not a Python
loop that unrolls into the jaxpr. The behavioural signature of that is *flatness*:
the top-level jaxpr equation count is invariant under the number of rounds (an
unrolled loop grows ~linearly with it). This is the deterministic stand-in for the
issue's "scan body compiles once, no unroll" -- stronger than a warm-time check and
not flaky. Byte-identity (the sumcheck / logup-gkr roundtrip tests) separately
proves the body genuinely runs R times: a hoisted or CSE'd carry would yield a
wrong proof.

Tracing only -- no execution -- so these run on every backend.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.challenge import ChallengePolicy
from zorch.sumcheck import verifier
from zorch.testkit.transcript import cheap_transcript
from zorch.verify import _scan_step, verify

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)


def _top_primitives(jaxpr: frx.core.ClosedJaxpr) -> list[str]:
    return [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]


def _verify_eqn_count(rounds: int) -> int:
    proof = fnp.ones((rounds, 2), KB)  # degree+1 = 2
    jaxpr = frx.make_jaxpr(
        lambda c, p, t: verify(verifier.SumcheckRound(1, challenges=_CH), c, p, t)
    )(fnp.array(0, KB), proof, cheap_transcript(KB))
    return len(jaxpr.jaxpr.eqns)


class VerifyScanShapeTest(absltest.TestCase):
    def test_verify_jaxpr_is_flat_in_trip_count(self) -> None:
        # Unrolled: eqn count grows with rounds. Scanned: one while-region, flat.
        self.assertEqual(_verify_eqn_count(3), _verify_eqn_count(7))

    def test_verify_lowers_to_a_scan(self) -> None:
        proof = fnp.ones((4, 2), KB)
        jaxpr = frx.make_jaxpr(
            lambda c, p, t: verify(verifier.SumcheckRound(1, challenges=_CH), c, p, t)
        )(fnp.array(0, KB), proof, cheap_transcript(KB))
        self.assertIn("scan", _top_primitives(jaxpr))


class ScanBodyIdentityTest(absltest.TestCase):
    """The scan body is memoized, so a warm trace cache survives repeated calls.

    `lax.scan` keys its cache on the identity of the function it is handed, and
    does not see through a `functools.partial`. A body built per call therefore
    re-traces an identical graph every time — invisible in results, ~285x the
    cost of the replay. Pinning identity is the deterministic form of "it does
    not recompile"; a timing assertion would be flaky and the scan exposes no
    cache-size handle for `testkit.jit_cache`.
    """

    def test_equal_rounds_share_one_scan_body(self) -> None:
        first = verifier.SumcheckRound(2, challenges=_CH)
        second = verifier.SumcheckRound(2, challenges=_CH)

        self.assertIsNot(first, second)
        self.assertIs(_scan_step(first), _scan_step(second))

    def test_rounds_that_trace_differently_get_their_own_body(self) -> None:
        degree_two = _scan_step(verifier.SumcheckRound(2, challenges=_CH))
        degree_three = _scan_step(verifier.SumcheckRound(3, challenges=_CH))

        self.assertIsNot(degree_two, degree_three)


if __name__ == "__main__":
    absltest.main()
