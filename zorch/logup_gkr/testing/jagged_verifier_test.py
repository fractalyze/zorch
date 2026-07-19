# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Prove -> verify roundtrip over a jagged GKR pyramid.

The chain runs a mixed schedule -- over-padded segments next to saturated
ones -- so every verifier layer replays round polynomials whose virtual-mass
corrections actually carried weight. The roundtrip closes against the input
layer's virtual dense leaf MLE, the check a PCS opening performs in
production.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    LogUpGkrOutput,
    _interleave,
    extract_jagged_outputs,
)
from zorch.logup_gkr.jagged_prover import JaggedGkrLayerRound, JaggedLayerProof
from zorch.logup_gkr.jagged_verifier import (
    JaggedGkrLayerRound as JaggedVerifierLayerRound,
)
from zorch.logup_gkr.prover import Carry, bind_output
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    random_jagged_layer,
    virtual_planes,
)
from zorch.poly.multilinear import eval_mle
from zorch.round import ProveChain, VerifyChain
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript, sample_challenge
from zorch.utils.bits import log2_strict_usize

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont

ROW_COUNTS = (3, 1, 5, 2)


def _bind_multi_limb(
    output: LogUpGkrOutput, transcript: Transcript, dtype: Any, limbs: int
) -> tuple[Carry, Transcript]:
    """`bind_output` with the multi-limb squeeze rule -- the binding glue a
    consumer whose claims extend the transcript's field owns."""
    num_vars = log2_strict_usize(output.numerator.shape[0])
    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(output.denominator)
    coords = []
    for _ in range(num_vars):
        transcript, c = sample_challenge(transcript, dtype, limbs)
        coords.append(c)
    point = fnp.stack(coords)
    num_eval = eval_mle(output.numerator, point)
    den_eval = eval_mle(output.denominator, point)
    return (num_eval, den_eval, point), transcript


def _prove(
    layers: list[JaggedGkrLayer],
) -> tuple[Carry, list[JaggedLayerProof], LogUpGkrOutput]:
    output = extract_jagged_outputs(layers[-1])
    carry, transcript = bind_output(output, cheap_transcript(KB))
    chain = ProveChain([JaggedGkrLayerRound(layer) for layer in reversed(layers[:-1])])
    final, _, proofs = chain(carry, transcript)
    return final, proofs, output


def _verify(
    output: LogUpGkrOutput, proofs: list[JaggedLayerProof]
) -> tuple[Carry, Array]:
    carry, transcript = bind_output(output, cheap_transcript(KB))
    chain = VerifyChain([JaggedVerifierLayerRound() for _ in proofs])
    final, _, ok = chain(carry, proofs, transcript)
    return final, ok


class JaggedRoundtripTest(absltest.TestCase):
    def test_roundtrip_accepts_and_closes_on_leaf_mle(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, ROW_COUNTS))
        prover_final, proofs, output = _prove(layers)
        self.assertEqual(len(proofs), len(layers) - 1)

        verifier_final, ok = _verify(output, proofs)
        self.assertTrue(bool(ok))
        # Fiat-Shamir lockstep: both sides reduce to the same point-claim.
        for got, want in zip(verifier_final, prover_final, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))

        # The PCS-consumer closing check, done directly: the reduced claims
        # are the input layer's interleaved virtual planes at the final point.
        num_eval, den_eval, point = verifier_final
        nrv = point.shape[0] - layers[0].num_batch_variables - 1
        n0, n1, d0, d1 = virtual_planes(layers[0], nrv)
        self.assertTrue(bool(num_eval == eval_mle(_interleave(n0, n1), point)))
        self.assertTrue(bool(den_eval == eval_mle(_interleave(d0, d1), point)))

    def test_tampered_round_poly_rejected(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(17, ROW_COUNTS))
        _, proofs, output = _prove(layers)
        bumped = proofs[1].round_polys.at[0, 0].add(fnp.array(1, KB))
        proofs[1] = replace(proofs[1], round_polys=bumped)
        _, ok = _verify(output, proofs)
        self.assertFalse(bool(ok))

    def test_tampered_opening_rejected(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(27, ROW_COUNTS))
        _, proofs, output = _prove(layers)
        bad = proofs[0].numerator_0 + fnp.array(1, KB)
        proofs[0] = replace(proofs[0], numerator_0=bad)
        _, ok = _verify(output, proofs)
        self.assertFalse(bool(ok))

    def test_multi_limb_roundtrip(self) -> None:
        # Extension-field claims over a base-field transcript: both sides
        # bind and squeeze through the same multi-limb rule.
        limbs = 4
        layers = build_jagged_pyramid(random_jagged_layer(37, ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])

        carry, transcript = _bind_multi_limb(output, cheap_transcript(KB), EF, limbs)
        chain = ProveChain(
            [JaggedGkrLayerRound(layer, limbs) for layer in reversed(layers[:-1])]
        )
        prover_final, _, proofs = chain(carry, transcript)

        carry, transcript = _bind_multi_limb(output, cheap_transcript(KB), EF, limbs)
        vchain = VerifyChain([JaggedVerifierLayerRound(limbs) for _ in proofs])
        verifier_final, _, ok = vchain(carry, proofs, transcript)
        self.assertTrue(bool(ok))
        self.assertEqual(verifier_final[0].dtype, EF)
        for got, want in zip(verifier_final, prover_final, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))


if __name__ == "__main__":
    absltest.main()
