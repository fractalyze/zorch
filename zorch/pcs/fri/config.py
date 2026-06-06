# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared FRI parameters, the code contract FRI requires, and proof types.

FRI is transparent, so the prover and verifier hold the *same* public params —
the degenerate case of the PCS prover-key/verifier-key split (no secret to keep
asymmetric). `FriParams` is that shared object. The query-phase machinery
(`LayerOpening`, the Fiat-Shamir position derivation) is scheme-neutral and
lives in `zorch.pcs.fold`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypeAlias, runtime_checkable

import jax
from jax import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.fold import LayerOpening

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
    jax.tree_util.register_dataclass,
    data_fields=["value", "layer_roots", "final_layer", "f_lo", "f_hi", "layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class FriProof:
    """value = claimed f(z); layer_roots = roots of committed fold layers
    1..num_rounds-1; final_layer = the last (cleartext, constant) codeword;
    f_lo/f_hi/layers carry the query openings, batched over the query axis.

    A registered pytree so it crosses the open/verify `@jit` boundary."""

    value: Array
    layer_roots: list[Array]
    final_layer: Array
    f_lo: Opening  # base-codeword conjugate pair, to rebuild the quotient
    f_hi: Opening
    layers: list[LayerOpening]  # committed layers 1 .. num_rounds-1
