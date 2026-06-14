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

    def pair_leaves(self, codeword: Array) -> Array:
        """Arrange a layer's codeword into its conjugate-pair leaves `[n//2, 2]`:
        row `p` is `(codeword[lo], codeword[hi])` for `(lo, hi) =
        pair_indices(p, level)`, so committing these leaves lets one Merkle path
        open both legs of the pair that folds to position `p`. The layout is the
        code's identity (natural order pairs a half-layer apart, bit-reversed
        order adjacently), so it lives behind the seam with `fold`."""
        ...

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened value pairs of layer `level` at `positions` (verifier side).

        `lo`/`hi` are the layer's entries at `pair_indices(positions, level)`.
        Must agree with `fold`: with `(l, h) = pair_indices(p, level)`,
        `fold(layer, beta)[p] == fold_values(layer[l], layer[h], beta, p, level)`.
        """
        ...

    def pair_indices(self, positions: Array, level: int) -> tuple[Array, Array]:
        """Leaf indices of layer `level`'s point pair for folded index
        `positions` — the pair whose fold lands at `positions` in layer
        `level + 1`. The pairing layout is part of the code's identity
        (natural order pairs a half-layer apart; bit-reversed order pairs
        adjacently), so it lives behind the seam with the fold."""
        ...

    def layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        """Per-layer folded query indices for sampled `positions`: element
        `i` addresses layer `i`'s pair via `pair_indices` and is where that
        fold lands in layer `i + 1`."""
        ...

    def check_final(self, final: Array, claim: Array) -> Array:
        """Whether the fully folded layer is the base-code encoding of the
        scalar `claim` — the IOPP terminal membership check, tied to the final
        sumcheck claim."""
        ...


@runtime_checkable
class KFoldableCode(LinearCode, Protocol):
    """k-ary generalization of the FoldableCode seam: a layer folds by a static
    `fold_factor` k, regrouping the k entries of one folded point's k-th-root
    coset per step instead of a conjugate pair (the k=2 special case).

    Additive to FoldableCode, not a replacement — a code may implement both
    (ReedSolomon does). The binary pair seam is left untouched so the k=2 path
    stays byte- and XLA-identical, while this k-group seam serves arbitrary-factor
    consumers. The two do not collapse into one: the single-group fold is Lagrange
    interpolation (`reed_solomon.fri_fold_k_values`), which at k=2 is the conjugate
    butterfly's strictly costlier twin, so they coexist by design (zorch#252 F1).
    Every method here is the k-ary mirror of a `FoldableCode` pair method, named
    `*group*` to sit alongside it."""

    fold_factor: int

    def fold_group(self, codeword: Array, beta: Array) -> Array:
        """Fold a layer by `beta`, dividing its length by `fold_factor` (prover
        side) — the k-ary counterpart of `FoldableCode.fold`."""
        ...

    def group_leaves(self, codeword: Array) -> Array:
        """Arrange a layer's codeword into its k-group leaves `[n // k, k]`: row
        `p` holds the `fold_factor` entries `group_indices(p, level)` whose fold
        lands at `p`, so one Merkle path opens the whole group. The k-ary
        `pair_leaves`."""
        ...

    def group_indices(self, positions: Array, level: int) -> tuple[Array, ...]:
        """The `fold_factor` leaf indices of layer `level`'s group whose fold
        lands at `positions` in layer `level + 1` — the k-ary `pair_indices`,
        returning k indices instead of a (lo, hi) pair."""
        ...

    def fold_group_values(
        self, group: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened k-groups of layer `level` at `positions` (verifier side).

        `group` is `[Q, k]`, the layer's entries at `group_indices(positions,
        level)`. Must agree with `fold_group`: with `g = group_indices(p, level)`,
        `fold_group(layer, beta)[p] == fold_group_values(layer[g], beta, p,
        level)`. The k-ary `fold_values`."""
        ...

    def group_layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        """Per-layer folded query indices for sampled `positions` under k-ary
        folding: element `i` addresses layer `i`'s group via `group_indices` and
        is where that fold lands in layer `i + 1`. The k-ary `layer_positions`."""
        ...

    def check_final(self, final: Array, claim: Array) -> Array:
        """Whether the fully folded layer is the base-code encoding of the scalar
        `claim` — identical to FoldableCode's terminal check (fold arity does not
        change the constant-polynomial membership test)."""
        ...
