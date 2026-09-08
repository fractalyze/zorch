# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Seam-level round trip: every PCS instance through ONE generic driver.

The per-scheme suites cover each instance's behavior; this test pins the
*interchangeability* claim. `_round_trip` is typed against the committer plus
the opening stage roles only, so each case below is simultaneously a runtime
commit→prove→verify round trip and a mypy check that the pair's wire types agree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.basefold.prover import BasefoldProver
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.pcs.fri.config import FriParams
from zorch.pcs.fri.prover import FriProver
from zorch.pcs.fri.verifier import FriVerifier
from zorch.pcs.kzg.prover import KzgProver
from zorch.pcs.kzg.testing.srs import toy_srs
from zorch.pcs.kzg.verifier import KzgVerifier
from zorch.pcs.stage import (
    CommittingOpener,
    OpeningClaim,
    OpeningProof,
    OpeningWitness,
)
from zorch.stage import TrivialClaim, VerifierStage
from zorch.testkit.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_ext_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import (
    DuplexTranscript,
    TranscriptT,
)

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont
SF = zk_dtypes.bn254_sf_mont

_GPU = frx.default_backend() == "gpu"

C = TypeVar("C")
D = TypeVar("D")
P = TypeVar("P")


def _round_trip(
    prover: CommittingOpener[C, D, P],
    verifier: VerifierStage[
        OpeningClaim[C], TrivialClaim, OpeningProof[P], TranscriptT
    ],
    polys: Sequence[Array],
    points: Sequence[Array],
    transcript: Callable[[], TranscriptT],
) -> Array:
    """commit → prove → verify through the seam types only."""
    commitment, prover_data = prover.commit(polys)
    claim = OpeningClaim(commitment, points)
    opened = prover.prove(claim, OpeningWitness(prover_data), transcript())
    return verifier.verify(claim, opened.reduction_proof, transcript()).ok


class SeamRoundTripTest(absltest.TestCase):
    def test_fri(self) -> None:
        _, _, tree = koalabear16_merkle()
        params = FriParams(
            code=ReedSolomon(message_len=4, blowup=2, dtype=KB),
            tree=tree,
            num_rounds=2,
            num_queries=3,
        )
        coeffs = fnp.array([1, 2, 3, 4], dtype=KB)
        z = fnp.array(2, dtype=KB)
        ok = _round_trip(
            FriProver(params),
            FriVerifier(params),
            [coeffs],
            [z],
            lambda: cheap_transcript(KB),
        )
        self.assertTrue(bool(ok))

    def test_basefold(self) -> None:
        log_s, S, K = 3, 8, 2
        rs = ReedSolomon(message_len=S, blowup=2, dtype=EF)
        _, _, tree = koalabear16_merkle()
        mle = rand_ext_field(1, (S, K), KB, EF)
        z = rand_ext_field(2, (log_s,), KB, EF)
        ok = _round_trip(
            BasefoldProver(rs, tree, num_queries=4),
            BasefoldVerifier(rs, tree, num_queries=4),
            [mle[:, k] for k in range(K)],
            [z],
            lambda: DuplexTranscript.new(koalabear16_perm(), rate=8),
        )
        self.assertTrue(bool(ok))

    @absltest.skipUnless(_GPU, "KZG commit/open use lax.msm, a GPU-only kernel")
    def test_kzg(self) -> None:
        pk, vk = toy_srs(tau=5, n=4)
        coeffs = fnp.array([3, 1, 4, 1], dtype=SF)
        z = fnp.array(7, dtype=SF)
        ok = _round_trip(
            KzgProver(pk),
            KzgVerifier(vk),
            [coeffs],
            [z],
            lambda: cheap_transcript(SF),
        )
        self.assertTrue(bool(ok))


if __name__ == "__main__":
    absltest.main()
