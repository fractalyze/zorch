# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The FoldableCode seam: a LinearCode whose codewords fold round-by-round.

BaseFold-style IOPPs (FRI, BaseFold) need more than linearity from their code.
Layer `level` (length `block_len >> level`; layer 0 is the fresh codeword)
pairs entries `(j, j + half)` as the two evaluations of a degree-1 polynomial
at a code-defined point pair, and the fold by challenge `beta` evaluates that
line at `beta`, halving the layer. That per-level point-pair structure —
`diag(T_i), diag(T'_i)` in BaseFold (Zeilberger–Chen–Fisch,
https://eprint.iacr.org/2023/1705, Definition 5) — is part of the code's
identity, not the PCS's: Reed-Solomon folds on the `(x, -x)` conjugates of its
NTT domain; a random foldable code folds on its sampled diagonals. Keeping the
fold behind this seam keeps the PCS layer free of encoding-specific domain
knowledge — a plain LinearCode (Brakedown, ...) is NOT enough to drive
BaseFold, so the prover/verifier must require this type, not LinearCode.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array

from zorch.coding.linear_code import LinearCode


@runtime_checkable
class FoldableCode(LinearCode, Protocol):
    def fold(self, codeword: Array, beta: Array) -> Array:
        """Fold a layer by `beta`, halving its length (prover side)."""
        ...

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened value pairs of layer `level` at `positions` (verifier side).

        `lo`/`hi` are the layer's entries at `positions` and
        `positions + (block_len >> (level + 1))`. Must agree with `fold`:
        `fold(layer, beta)[p] ==
        fold_values(layer[p], layer[p + half], beta, p, level)`.
        """
        ...

    def check_final(self, final: Array, claim: Array) -> Array:
        """Whether the fully folded layer is the base-code encoding of the
        scalar `claim` — the IOPP terminal membership check, tied to the final
        sumcheck claim."""
        ...
