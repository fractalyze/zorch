# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed root and checks
the FRI fold consistency plus the jagged opening sumchecks. It holds only the
public params (`rs` for the domain/blowup, `tree` for the Merkle config) — never
the prover's retained codeword. The body lands in P3/P4 alongside `open`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.transcript import Transcript


@dataclass(frozen=True)
class BasefoldVerifier:
    """BaseFold PCS verifier (`PcsVerifier`)."""

    rs: ReedSolomon
    tree: MerkleTree
    # Must match the prover's; placeholder count, not soundness-calibrated.
    num_queries: int = 4

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: Any,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        raise NotImplementedError("BasefoldVerifier.verify lands in P3/P4")
