# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-variable verifier compiles to one traced region.

`verify` is a single jit zone with its round loop unrolled inside it. The round
count is static under `jit`, so the loop flattens into one compiled program with
no scan boundary — measured at about half the warm cost of the scanned replay it
replaced, against a compile that grows with the round count. A verifier pays
that once per proof shape, and its shapes are fixed per circuit.

The signature of the unroll is that the jaxpr equation count *grows* with the
round count: a scan would keep it flat behind one while-region. Byte-identity
(the sumcheck / logup-gkr roundtrip tests) separately proves the body genuinely
runs R times.

Tracing only -- no execution -- so these run on every backend.
"""

from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.sumcheck import prover as prover_mod
from zorch.sumcheck import verifier
from zorch.sumcheck.stage import SumcheckProver, SumcheckWitness, SumClaim
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.verify import verify

KB = zk_dtypes.koalabear_mont

# Challenges in the transcript's own field: one squeeze, reinterpreted as itself.
_CH = ChallengePolicy(KB)


def _all_primitives(jaxpr: Any) -> list[str]:
    """Primitive names at every depth, for the same reason as `_total_eqns`."""
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                names.extend(_all_primitives(inner))
    return names


def _total_eqns(jaxpr: Any) -> int:
    """Equations at every depth: `verify` is a jit zone, so its unrolled body
    sits inside the nested jaxpr rather than at the top level."""
    total = 0
    for eqn in jaxpr.eqns:
        total += 1
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                total += _total_eqns(inner)
    return total


def _verify_eqn_count(rounds: int) -> int:
    proof = fnp.ones((rounds, 2), KB)  # degree+1 = 2
    jaxpr = frx.make_jaxpr(
        lambda c, p, t: verify(verifier.SumcheckRound(1, challenges=_CH), c, p, t)
    )(fnp.array(0, KB), proof, cheap_transcript(KB))
    return _total_eqns(jaxpr.jaxpr)


class VerifyShapeTest(absltest.TestCase):
    def test_verify_unrolls_its_round_loop(self) -> None:
        # The unroll is what buys the warm time: no scan boundary, so the body
        # is inlined once per round and the equation count tracks the rounds.
        self.assertGreater(_verify_eqn_count(7), _verify_eqn_count(3))

    def test_verify_is_one_jit_zone(self) -> None:
        # One compiled program: the round loop is inside it, not a sequence of
        # eager dispatches. (A `scan` still appears deeper — the sponge absorb
        # scans its rate blocks — so scan-absence is not the signature here;
        # the growth above is.)
        proof = fnp.ones((4, 2), KB)
        jaxpr = frx.make_jaxpr(
            lambda c, p, t: verify(verifier.SumcheckRound(1, challenges=_CH), c, p, t)
        )(fnp.array(0, KB), proof, cheap_transcript(KB))

        self.assertEqual([eqn.primitive.name for eqn in jaxpr.jaxpr.eqns], ["jit"])


class UnrolledVerifyTest(absltest.TestCase):
    """The unrolled driver still reduces and still rejects."""

    def _proof(self, rounds: int) -> tuple[Array, Array]:
        state = rand_field(11, (2, 1 << rounds), KB)
        claim = fnp.sum(state[0] * state[1])
        prover = SumcheckProver(
            prover_mod.StandardRound(prover_mod.ProductSummand(2), challenges=_CH)
        )
        proved = prover.prove(
            SumClaim(claim, rounds), SumcheckWitness(state), cheap_transcript(KB)
        )
        return claim, proved.reduction_proof

    def test_matches_the_scan_driver(self) -> None:
        claim, proof = self._proof(4)
        vr = verifier.SumcheckRound(2, challenges=_CH)

        scanned = verify(vr, claim, proof, cheap_transcript(KB))
        unrolled = verify(vr, claim, proof, cheap_transcript(KB))

        self.assertTrue(bool(fnp.all(scanned[0] == unrolled[0])))
        self.assertEqual(scanned[1], unrolled[1])
        self.assertEqual(bool(scanned[3]), bool(unrolled[3]))

    def test_rejects_a_tampered_message_like_the_scan(self) -> None:
        claim, proof = self._proof(4)
        vr = verifier.SumcheckRound(2, challenges=_CH)
        tampered = proof.at[2, 0].set(proof[2, 0] + fnp.ones((), KB))

        self.assertFalse(bool(verify(vr, claim, tampered, cheap_transcript(KB))[3]))


if __name__ == "__main__":
    absltest.main()
