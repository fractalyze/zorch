# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold proof type. Co-located with the scheme like `FriProof` in
`fri/config.py` (kzg needs no proof dataclass) — there is no `proof.py` in
`pcs/`, so this keeps the three schemes consistent. The staggered batch weights
both `open` and `verify` derive live in `pcs/basefold/batching.py`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import jax
from jax import Array

from zorch.commit.merkle import Opening

# A single Merkle root binding the whole column batch (the matrix commitment).
BasefoldCommitment: TypeAlias = Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "univariate_messages",
        "fri_roots",
        "final_poly",
        "component_openings",
        "query_openings",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class BasefoldProof:
    """One BaseFold batch-open proof: several separately committed matrices
    reduced to one FRI by a staggered partial-Lagrange RLC.

    univariate_messages: per fold round, the degree-1 sumcheck message
        `(zero_val, one_val)` = `(s(0), s(1))`.
    fri_roots: pair-leaf commitment roots of the batched codeword, one per fold
        round (the round commits the *pre-fold* layer's conjugate pairs, then
        folds). Length `num_vars`.
    final_poly: the final folded batched codeword (a constant on the order-`blowup`
        domain), cleartext; the IOPP terminal membership check ties it to the
        final claim.
    component_openings: one `Opening` per committed matrix, opened at the shared
        query positions (`row` is that matrix's `[Q, width]` columns); the
        verifier staggered-RLCs them into the batched value at each position.
    query_openings: one `Opening` per fold round, the batched codeword's opened
        pair-leaf (`row` is `[Q, 2]`).
    """

    univariate_messages: list[tuple[Array, Array]]
    fri_roots: list[Array]
    final_poly: Array
    component_openings: list[Opening]
    query_openings: list[Opening]
