# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Single-shot Ligero proof type. Co-located with the scheme like `BasefoldProof`
in `basefold/config.py` — there is no shared `proof.py` in `pcs/`.

Ligero opens one committed multilinear `f`, laid out as a `rows x cols` matrix
`X̃` (`rows = code.message_len`), at a point `z = (z_row, z_col)`. The prover
sends `w = X̃ · r_col` in the clear (`r_col = eq(z_col)`); the verifier checks
proximity `<X[s], r_col> = encode(w)[s]` on a few opened codeword rows and value
`<r_row, w> = y`. That single sent vector is Ligero's `sqrt(N)` proof cost — the
recursion in `pcs/ligerito` is exactly what removes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import jax
from jax import Array

from zorch.commit.merkle import Opening

# A single Merkle root binding the codeword rows (the matrix commitment).
LigeroCommitment: TypeAlias = Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["w", "component_opening"],
    meta_fields=[],
)
@dataclass(frozen=True)
class LigeroProof:
    """One single-shot Ligero opening proof.

    w: the sent vector `X̃ · r_col`, length `rows` (`= code.message_len`). Its
        `sqrt(N)` size is Ligero's proof-cost bottleneck.
    component_opening: the codeword rows opened at the sampled query positions
        (`row` is `[Q, cols]`); the verifier dots each with `r_col` for the
        proximity left-hand side and Merkle-checks it against the commitment.
    """

    w: Array
    component_opening: Opening
