# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold prover — the commit slice of the multilinear PCS on the `pcs` seam.

`commit` is the RS low-degree extension (the native NTT via `ReedSolomon`) of each
column followed by a Merkle commit of the codeword rows. Unlike `kzg`/`fri` — which
commit each polynomial in the batch independently and return one root per poly —
BaseFold is a **matrix commitment**: the columns share one RS domain and the
Merkle leaves are codeword *rows* spanning all columns, so the whole batch binds
under a single root. That single-root commitment is what the jagged structure bind
(`JaggedPcs`) hashes against. Both halves are already-fused substrate ops, so the
commit is one `@jit`-able device zone with no host sync. `open` (the FRI query
phase + the jagged opening sumchecks) lands in P3.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.transcript import Transcript


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["digest_layers"],
    meta_fields=["widths"],
)
@dataclass(frozen=True)
class BasefoldProverData:
    """Retained witness from `BasefoldProver.commit`: the Merkle digest layers over
    the RS codeword, plus per-column widths. Consumed by `open` (P3).

    A pytree (digest_layers = leaves, widths = static meta) so a `commit` call can
    return it from inside a `@jit` zone.
    """

    digest_layers: list[Array]
    # len-1 today (one MLE per commit); reserved for batch-commit in P3.
    widths: tuple[int, ...]


@dataclass(frozen=True)
class BasefoldProver:
    """BaseFold PCS prover (`PcsProver`). `rs` fixes the per-column message length
    (= the MLE height `S`); `tree` commits the codeword rows."""

    rs: ReedSolomon
    tree: MerkleTree

    def commit(self, polys: Sequence[Array]) -> tuple[Array, BasefoldProverData]:
        # The columns share one RS message length S, so stack them into the [S, K]
        # matrix and encode column-wise. ReedSolomon.encode is last-axis, hence the
        # transpose in/out: [S,K] -> [K,S] -> encode -> [K, S*blowup] -> [S*blowup, K].
        mle = jnp.stack(polys, axis=1)
        codeword = self.rs.encode(mle.T).T
        root, layers = self.tree.commit(codeword)
        return root, BasefoldProverData(digest_layers=layers, widths=(len(polys),))

    def open(
        self,
        prover_data: BasefoldProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, Any, Transcript]:
        raise NotImplementedError("BasefoldProver.open lands in P3 (jagged opening)")
