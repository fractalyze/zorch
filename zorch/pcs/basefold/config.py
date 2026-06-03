# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold proof type. Co-located with the scheme like `FriProof` in
`fri/config.py` (kzg needs no proof dataclass) — there is no `proof.py` in
`pcs/`, so this keeps the three schemes consistent."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
from jax import Array

from zorch.pcs.fri.config import LayerOpening


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "univariate_messages",
        "fri_roots",
        "final_poly",
        "component_opening",
        "query_openings",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class BasefoldProof:
    """One BaseFold opening proof.

    univariate_messages: per fold round, the degree-1 sumcheck message
        `(zero_val, one_val)` = `(s(0), s(1))`.
    fri_roots: Merkle roots of the committed fold layers 1..num_vars-1
        (layer 0 is the original matrix commitment, not recommitted).
    final_poly: the final (constant) codeword after all folds, length
        `block_len >> num_vars` (= `blowup` when opening over all `log S` vars).
    component_opening: conjugate-pair openings of the ORIGINAL `[n, K]` matrix
        at the layer-0 query indices (verifier RLC-combines the K columns).
    query_openings: conjugate-pair openings of each committed fold layer.
    """

    univariate_messages: list[tuple[Array, Array]]
    fri_roots: list[Array]
    final_poly: Array
    component_opening: LayerOpening
    query_openings: list[LayerOpening]
