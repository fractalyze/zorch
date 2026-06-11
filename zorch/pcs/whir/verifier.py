# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` replays the WHIR rounds from the transcript: it re-derives the folding,
out-of-domain and query challenges in the prover's order, checks each round's
degree-2 sumcheck message against the running claim, rebuilds the strided query
openings from the committed roots, folds each opened `2^k_whir` coset as a small
MLE at the round's folding challenges (the binary k-fold consistency), and closes
on the final-poly constraint. It holds only the public params (`code` for the RS
geometry, `tree` for the Merkle config) — never the prover's retained codeword.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.pcs.whir.config import WhirCommitment, WhirParams, WhirProof
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class WhirVerifier:
    """WHIR PCS verifier (`PcsVerifier`). `code`/`tree`/`params` must match the
    prover's."""

    code: ReedSolomon
    tree: StridedMerkleTree
    params: WhirParams

    def verify(
        self,
        commitment: WhirCommitment,
        points: Sequence[Array],
        values: Array,
        proof: WhirProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        raise NotImplementedError("WhirVerifier.verify — Task 6")


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[WhirCommitment, WhirProof]] = WhirVerifier
