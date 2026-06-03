# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-variable drivers compile to one traced region, flat in trip count.

Issue #58: `prove` / `verify` must lower their round loop to a single `lax.scan`,
not a Python loop that unrolls into the jaxpr. The behavioural signature of that
is *flatness*: the top-level jaxpr equation count is invariant under the number of
rounds (an unrolled loop grows ~linearly with it). This is the deterministic
stand-in for the issue's "scan body compiles once, no unroll" -- stronger than a
warm-time check and not flaky. Byte-identity (the sumcheck / logup-gkr roundtrip
tests) separately proves the body genuinely runs R times: a hoisted or CSE'd carry
would yield a wrong proof.

Tracing only -- no execution -- so these run on every backend (the zkx#500 CPU
while-emitter bug only bites at run time; see transcript_test).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.prove import prove
from zorch.sumcheck import prover, verifier
from zorch.transcript import StubTranscript
from zorch.verify import verify

KB = zk_dtypes.koalabear


def _top_primitives(jaxpr: jax.core.ClosedJaxpr) -> list[str]:
    return [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]


def _verify_eqn_count(rounds: int) -> int:
    proof = jnp.ones((rounds, 2), KB)  # degree+1 = 2
    jaxpr = jax.make_jaxpr(lambda c, p, t: verify(verifier.SumcheckRound(1), c, p, t))(
        jnp.array(0, KB), proof, StubTranscript(jnp.zeros(rounds, KB))
    )
    return len(jaxpr.jaxpr.eqns)


def _prove_eqn_count(rounds: int) -> int:
    f = jnp.arange(1, (1 << rounds) + 1, dtype=KB)
    jaxpr = jax.make_jaxpr(lambda s, t: prove(prover.SumcheckRound(1), s, t))(
        [f], StubTranscript(jnp.zeros(rounds, KB))
    )
    return len(jaxpr.jaxpr.eqns)


class VerifyScanShapeTest(absltest.TestCase):
    def test_verify_jaxpr_is_flat_in_trip_count(self) -> None:
        # Unrolled: eqn count grows with rounds. Scanned: one while-region, flat.
        self.assertEqual(_verify_eqn_count(3), _verify_eqn_count(7))

    def test_verify_lowers_to_a_scan(self) -> None:
        proof = jnp.ones((4, 2), KB)
        jaxpr = jax.make_jaxpr(
            lambda c, p, t: verify(verifier.SumcheckRound(1), c, p, t)
        )(jnp.array(0, KB), proof, StubTranscript(jnp.zeros(4, KB)))
        self.assertIn("scan", _top_primitives(jaxpr))


class ProveScanShapeTest(absltest.TestCase):
    def test_prove_jaxpr_is_flat_in_trip_count(self) -> None:
        # 8 vs 32 leaves -> 3 vs 5 rounds; a fixed-width scan body is flat.
        self.assertEqual(_prove_eqn_count(3), _prove_eqn_count(5))

    def test_prove_lowers_to_a_scan(self) -> None:
        f = jnp.arange(1, 17, dtype=KB)
        jaxpr = jax.make_jaxpr(lambda s, t: prove(prover.SumcheckRound(1), s, t))(
            [f], StubTranscript(jnp.zeros(4, KB))
        )
        self.assertIn("scan", _top_primitives(jaxpr))


if __name__ == "__main__":
    absltest.main()
