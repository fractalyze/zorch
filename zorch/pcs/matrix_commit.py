# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared matrix-commit core for the row-leaf PCS family (Ligero, Ligerito).

Each of these schemes lays a witness out as a `[batch, message_len]` matrix,
low-degree-extends it along the message axis with a `LinearCode`, reinterprets
the codeword's rows as base-field Merkle leaves, and commits them. Only the
*retained* artifacts differ per scheme (Ligero also keeps the message matrix,
Ligerito keeps just the slim commitment), so the commit *computation* lives here
and each prover keeps what it needs from the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import frx
from frx import Array

from zorch.coding.linear_code import LinearCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.fold import to_base_field


def _identity(matrix: Array) -> Array:
    return matrix


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["root", "leaves", "digest_layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class CommittedMatrix:
    """One matrix commitment's retained artifacts: the root, the base-field
    Merkle leaves `[block_len, batch*limbs]`, and the digest layers. A registered
    pytree so it crosses a committing `@jit` boundary. Schemes that fold the
    codeword or reopen the message matrix keep those alongside — this is the
    Merkle-commit core they share."""

    root: Array
    leaves: Array
    digest_layers: list[Array]


def commit_matrix(
    code: LinearCode,
    tree: MerkleTree,
    message_matrix: Array,
    *,
    pre: Callable[[Array], Array] = _identity,
) -> CommittedMatrix:
    """Encode `pre(message_matrix)` (`[batch, message_len]`) along its message
    axis, reinterpret the `[batch, block_len]` codeword's rows as base-field
    Merkle leaves `[block_len, batch*limbs]`, and commit.

    Only the leaf orientation is returned. This used to hand back the raw
    codeword alongside it, speculatively, "for schemes that fold it" — no such
    scheme ever arrived and both callers discarded it. A scheme that needs the
    codeword should encode it itself; re-add a second return only with a
    consumer in the same change."""
    codeword = code.encode(pre(message_matrix))  # [batch, block_len]
    leaves = to_base_field(codeword.T)  # [block_len, batch*limbs]
    root, layers = tree.commit(leaves)
    return CommittedMatrix(root, leaves, layers)
