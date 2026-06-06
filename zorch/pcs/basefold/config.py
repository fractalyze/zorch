# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold proof type and the shared RLC-batching helper. Co-located with the
scheme like `FriProof` in `fri/config.py` (kzg needs no proof dataclass) — there
is no `proof.py` in `pcs/`, so this keeps the three schemes consistent. The
`sample_rlc_coeffs` helper lives here so `open` and `verify` derive the column
weights from one definition and cannot drift, the same way `fri/config` shares
its query-index derivation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

from zorch.pcs.fold import LayerOpening
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize

# A single Merkle root binding the whole column batch (the matrix commitment).
BasefoldCommitment: TypeAlias = Array


def sample_rlc_coeffs(
    transcript: Transcript, k: int, dtype: Any
) -> tuple[Transcript, Array]:
    """RLC weights `(k,)` that batch `k` columns into one codeword, from
    `log2_ceil(k)` squeezed challenges (`k == 1` -> `[1]`, no squeeze). Called by
    both `open` and `verify` so the two sides sample identically."""
    nbv = log2_ceil_usize(k)
    if nbv == 0:
        return transcript, jnp.ones(1, dtype)
    transcript, s = transcript.sample(nbv)
    return transcript, expand_eq_to_hypercube(s, jnp.ones((), dtype))[:k]


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
