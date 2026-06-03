# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold multilinear PCS — P2 commit slice.

`commit` is the RS low-degree extension (the native NTT via `ReedSolomon`) of
each MLE column followed by a Merkle commit of the codeword rows. Both halves
are already-fused substrate ops, so the whole commit is one `@jit`-able device
zone with no host sync. `open` / `verify` (the FRI query phase + the jagged
opening sumchecks) land in P3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasefoldProverData:
    """Retained witness from `Basefold.commit`: the Merkle digest layers over
    the RS codeword, plus per-MLE column widths. Consumed by `open` (P3).

    Registered as a pytree (digest_layers = children, widths = static aux) so a
    `commit` call can return it from inside a `@jit` zone.
    """

    digest_layers: list[Array]
    widths: tuple[
        int, ...
    ]  # len-1 today (one MLE per commit); reserved for batch-commit in P3.

    def tree_flatten(self) -> tuple[tuple[list[Array]], tuple[int, ...]]:
        return (self.digest_layers,), self.widths

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[int, ...], children: tuple[list[Array]]
    ) -> "BasefoldProverData":
        return cls(digest_layers=children[0], widths=aux)


class Basefold:
    """BaseFold PCS (`Pcs`). `rs` fixes the per-column message length (= the MLE
    height `S`); `tree` commits the codeword rows."""

    def __init__(self, rs: ReedSolomon, tree: MerkleTree) -> None:
        self.rs = rs
        self.tree = tree

    def commit(self, mle: Array) -> tuple[Array, BasefoldProverData]:
        # Each column of mle is one MLE poly sharing RS message length S, so
        # encode column-wise. ReedSolomon.encode is last-axis, hence transpose
        # in/out: [S,K] -> [K,S] -> encode -> [K, S*blowup] -> [S*blowup, K].
        codeword = self.rs.encode(mle.T).T
        root, layers = self.tree.commit(codeword)
        return root, BasefoldProverData(digest_layers=layers, widths=(mle.shape[1],))

    def open(
        self, prover_data: Any, point: Array, transcript: Any
    ) -> tuple[Array, Any]:
        raise NotImplementedError("Basefold.open lands in P3 (jagged opening)")

    def verify(
        self,
        commitment: Array,
        point: Array,
        value: Array,
        proof: Any,
        transcript: Any,
    ) -> bool:
        raise NotImplementedError("Basefold.verify lands in P3/P4")
