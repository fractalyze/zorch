# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared FRI parameters, the code contract FRI requires, and proof types.

FRI is transparent, so the prover and verifier hold the *same* public params —
the degenerate case of the PCS prover-key/verifier-key split (no secret to keep
asymmetric). `FriParams` is that shared object. The query-phase machinery
(the pre-fold pair-leaf round, the Fiat-Shamir position derivation) is
scheme-neutral and lives in `zorch.pcs.fold`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypeAlias, runtime_checkable

import frx
from frx import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree, Opening

FriCommitment: TypeAlias = Array  # stacked Merkle roots, one per committed poly


@runtime_checkable
class DeepFoldableCode(FoldableCode, Protocol):
    """A FoldableCode whose codewords are evaluations on a queryable domain.

    The DEEP quotient `(f(x) − v)/(x − z)` divides pointwise by the layer-0
    domain coordinates, so FRI needs `domain()` on top of the fold contract —
    a fold-only code with no evaluation-domain notion cannot drive it."""

    def domain(self) -> Array:
        """The layer-0 evaluation points, coset shift included."""
        ...


@dataclass(frozen=True)
class FriParams:
    """Public FRI configuration, identical on both sides."""

    code: DeepFoldableCode  # LDE; the fold seam + the DEEP quotient's domain
    tree: MerkleTree  # Merkle commitment over codeword leaves
    num_rounds: int  # fold rounds; final codeword has block_len >> num_rounds entries
    num_queries: int  # query repetitions (soundness amplification)


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["value", "fri_roots", "final_layer", "f_opening", "query_openings"],
    meta_fields=[],
)
@dataclass(frozen=True)
class FriProof:
    """value = claimed f(z); fri_roots = pair-leaf commitment roots of the
    quotient's fold layers 0..num_rounds-1 (the round commits the *pre-fold*
    layer's conjugate pairs, then folds); final_layer = the last (cleartext,
    constant) codeword; f_opening = `f`'s conjugate-pair leaf, used to rebuild the
    quotient at layer 0; query_openings = each fold layer's opened pair-leaf.
    All openings are batched over the query axis.

    A registered pytree so it crosses the open/verify `@jit` boundary."""

    value: Array
    fri_roots: list[Array]
    final_layer: Array
    f_opening: Opening  # base-codeword conjugate-pair leaf (row [Q, 2])
    query_openings: list[Opening]  # committed fold layers 0 .. num_rounds-1
