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

from zorch.sumcheck import verifier
from zorch.testkit.transcript import cheap_transcript
from zorch.verify import verify

KB = zk_dtypes.koalabear_mont


def _top_primitives(jaxpr: frx.core.ClosedJaxpr) -> list[str]:
    return [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]


def _verify_eqn_count(rounds: int) -> int:
    proof = fnp.ones((rounds, 2), KB)  # degree+1 = 2
    jaxpr = frx.make_jaxpr(lambda c, p, t: verify(verifier.SumcheckRound(1), c, p, t))(
        fnp.array(0, KB), proof, cheap_transcript(KB)
    )
    return len(jaxpr.jaxpr.eqns)


class VerifyScanShapeTest(absltest.TestCase):
    def test_verify_jaxpr_is_flat_in_trip_count(self) -> None:
        # Unrolled: eqn count grows with rounds. Scanned: one while-region, flat.
        self.assertEqual(_verify_eqn_count(3), _verify_eqn_count(7))

    def test_verify_lowers_to_a_scan(self) -> None:
        proof = fnp.ones((4, 2), KB)
        jaxpr = frx.make_jaxpr(
            lambda c, p, t: verify(verifier.SumcheckRound(1), c, p, t)
        )(fnp.array(0, KB), proof, cheap_transcript(KB))
        self.assertIn("scan", _top_primitives(jaxpr))


if __name__ == "__main__":
    absltest.main()
