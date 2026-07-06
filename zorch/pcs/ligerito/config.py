# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito recursive-open schedule + proof type. Co-located with the scheme
like `LigeroProof` / `BasefoldProof` — there is no shared `proof.py` in `pcs/`.

Ligerito removes single-shot Ligero's `sqrt(N)` sent vector `w` by *committing*
`w` instead of sending it, then discharging the proximity check as one continuous
interleaved sumcheck that batches every level's committed-`w` eval-claims and
recurses on the residual. The schedule below fixes, per
recursive level, how many witness variables the sumcheck folds before re-committing
and at what (shrinking) rate. Code-generic over a `TensorCode`; the multiplicative
Reed-Solomon instantiation is the de-risk vehicle (fractalyze/flock-zorch#32),
the additive-NTT (GHASH) one is #11/#27.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TypeAlias

import jax
from jax import Array

from zorch.commit.merkle import Opening

# The level-0 Merkle root binding the initial Ligero matrix commitment. The
# recursive levels' re-commit roots ride in the proof, not here.
LigeritoCommitment: TypeAlias = Array


@dataclass(frozen=True)
class LigeritoConfig:
    """The per-level schedule of a Ligerito recursive open.

    A level `i` folds `fold_ks[i]` witness variables through the continuous
    interleaved sumcheck, then — unless it is the final (residual) level —
    re-commits the folded witness as a fresh Ligero matrix at inverse rate
    `2^log_inv_rates[i]` and opens `queries[i]` codeword rows for that level's
    proximity check. Lowering the rate each level shrinks the query count while
    the witness shrinks. The committed multilinear has
    `num_vars` variables; the final level sends the remaining `residual_vars`
    (`= num_vars - sum(fold_ks)`) in the clear.

    num_vars: the committed multilinear's variable count. Must be
        `>= sum(fold_ks)`; the excess is the plaintext residual.
    fold_ks: variables folded per level.
    log_inv_rates: inverse-rate log2 of each *committed* matrix — the initial
        commit plus one per non-final level. `len == len(fold_ks)` (index `j` is
        `M_j`'s rate; see the prover for the exact map).
    queries: opened-row count per committed matrix; `len == len(fold_ks)`.
        Placeholder counts, not soundness-calibrated (like Ligero/BaseFold).
    ood_samples: out-of-domain binding samples per level; `()` means no OOD (the
        RS de-risk gate default — flock uses OOD for soundness, calibrated later).
    """

    num_vars: int
    fold_ks: tuple[int, ...]
    log_inv_rates: tuple[int, ...]
    queries: tuple[int, ...]
    ood_samples: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.fold_ks:
            raise ValueError("fold_ks must be non-empty; Ligerito needs >= 1 level")
        if any(k <= 0 for k in self.fold_ks):
            raise ValueError(
                f"every fold_ks entry must be positive; got {self.fold_ks}"
            )
        # residual >= 1: a 0-var residual would make the last committed matrix
        # message-length 1 (no encoding, its proximity check vacuous) — over-fold.
        if self.num_vars <= sum(self.fold_ks):
            raise ValueError(
                f"num_vars={self.num_vars} must be > sum(fold_ks)="
                f"{sum(self.fold_ks)} (the residual must carry >= 1 variable)"
            )
        if not len(self.fold_ks) == len(self.log_inv_rates) == len(self.queries):
            raise ValueError(
                "fold_ks, log_inv_rates, queries must be the same length; got "
                f"{len(self.fold_ks)}/{len(self.log_inv_rates)}/{len(self.queries)}"
            )

    @property
    def num_levels(self) -> int:
        return len(self.fold_ks)

    @property
    def residual_vars(self) -> int:
        return self.num_vars - sum(self.fold_ks)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "sumcheck_messages",
        "recursive_roots",
        "component_openings",
        "final_residual",
        "ood_values",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class LigeritoProof:
    """One Ligerito recursive-open proof.

    NOTE: first-draft wire, co-evolving with the prover/verifier (fractalyze/
    flock-zorch#32 slice 1) — fields may still move as the round-trip settles.

    sumcheck_messages: the continuous interleaved sumcheck's per-variable
        messages (degree-2 product `ŵ·B`, so each carries the round polynomial's
        non-reconstructable evals). One per bound variable across all levels.
    recursive_roots: the re-commit Merkle root of each non-initial committed
        level (the initial root is the `LigeritoCommitment`, passed to `verify`).
    component_openings: opened codeword rows per committed level, at that level's
        sampled query positions — the proximity left-hand side `<X[s], r_col>`.
    final_residual: the plaintext folded witness of the final level; the verifier
        replays the batched sumcheck's terminal claim against it.
    ood_values: out-of-domain claim values (empty when `ood_samples` is `()`).
    """

    sumcheck_messages: list[Array]
    recursive_roots: list[Array]
    component_openings: list[Opening]
    final_residual: Array
    ood_values: list[Array] = field(default_factory=list)
