# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared DEEP-FRI parameters, the code contract the scheme requires, and the
proof type.

DEEP-FRI is transparent, so the prover and verifier hold the *same* public
params — the degenerate case of the PCS prover-key/verifier-key split (no
secret to keep asymmetric). `DeepFriParams` is that shared object. The batched
quotient arithmetic is `zorch.pcs.deep`'s and the fold/query machinery (the
pre-fold pair-leaf round, the Fiat-Shamir position derivation) is
`zorch.pcs.fold`'s; both are scheme-neutral, so this package holds only what is
DEEP-FRI's own — the stage roles and their wire types.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypeAlias, runtime_checkable

import frx
from frx import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree, Opening

DeepFriCommitment: TypeAlias = Array  # stacked Merkle roots, one per committed poly


@runtime_checkable
class DeepFoldableCode(FoldableCode, Protocol):
    """A FoldableCode whose codewords are evaluations on a queryable domain.

    The DEEP quotient `(f(x) − v)/(x − z)` divides pointwise by the layer-0
    domain coordinates, so DEEP-FRI needs `domain()` on top of the fold
    contract — a fold-only code with no evaluation-domain notion cannot drive
    it."""

    def domain(self) -> Array:
        """The layer-0 evaluation points, coset shift included."""
        ...


@dataclass(frozen=True)
class DeepFriParams:
    """Public DEEP-FRI configuration, identical on both sides.

    The two degenerate schedules are rejected at construction rather than
    downstream: zero rounds satisfies the verifier's structural length guard
    and then crashes its layer-0 rebuild, and zero queries makes every Merkle,
    composition, and fold check vacuously pass — a verifier that accepts
    anything. Everything else the schedule needs is checked where it is used.
    """

    code: DeepFoldableCode  # LDE; the fold seam + the DEEP quotients' domain
    tree: MerkleTree  # Merkle commitment over codeword leaves
    num_rounds: int  # fold rounds; final codeword has block_len >> num_rounds entries
    num_queries: int  # query repetitions (soundness amplification)

    def __post_init__(self) -> None:
        if self.num_queries < 1:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}")
        max_rounds = self.code.block_len.bit_length() - 1
        if not 1 <= self.num_rounds <= max_rounds:
            raise ValueError(
                f"num_rounds must be in [1, {max_rounds}] (the code folds "
                f"block_len {self.code.block_len} at most that far), got "
                f"{self.num_rounds}"
            )


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["fri_roots", "final_layer", "f_openings", "query_openings"],
    meta_fields=[],
)
@dataclass(frozen=True)
class DeepFriProof:
    """One proof for the whole batch — the single fold chain covering the
    batched DEEP composition. fri_roots = pair-leaf commitment roots of the
    composition's fold layers 0..num_rounds-1 (the round commits the *pre-fold*
    layer's conjugate pairs, then folds); final_layer = the last (cleartext,
    constant) codeword; f_openings = each committed polynomial's conjugate-pair
    leaf at the layer-0 query positions, from which the verifier rebuilds the
    batched quotient; query_openings = each fold layer's opened pair-leaf. All
    openings are batched over the query axis. The claimed evaluations ride
    `OpeningProof.values`, not this proof — the seam already carries them once.

    A registered pytree so it crosses the open/verify `@jit` boundary."""

    fri_roots: list[Array]
    final_layer: Array
    f_openings: list[Opening]  # per committed poly: pair leaf (row [Q, 2]) at a[0]
    query_openings: list[Opening]  # committed fold layers 0 .. num_rounds-1
