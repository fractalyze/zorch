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
    ood_samples: out-of-domain binding blocks per recursive commit:
        `ood_samples[j]` blocks run right after M_{j+1}'s root is observed (the
        end of level `j`), each sampling a fresh point `z_ood` of the folded
        witness's arity, gluing its eval-claim `(eq(z_ood), W(z_ood))` into the
        running sumcheck with a separation challenge — binding the just-made
        commitment at a point outside the query domain. `len == num_levels - 1`
        (the final level commits nothing), or `()` for none (the RS de-risk
        default — flock uses OOD for soundness, calibrated later).
    alpha_lsb_first: index orientation of the per-level partial-Lagrange batching
        weights (the query-claim glue's alpha). False = the native MSB-first
        expansion; True = LSB-first (challenge `j` <-> table bit `j`), for wire
        formats that fix that convention. Prover and verifier both derive from it,
        so the round trip holds either way; it only changes the produced bytes.
    compressed_sumcheck_messages: round-message wire form. False = the natural
        domain evals `[s(0), s(1), s(2)]` (`SumcheckRound(degree=2)`); True = the
        compressed coefficients `[c_0, c_2]` with the linear coefficient
        reconstructed from the running claim (`CompressedProductRound`). Another
        wire-convention knob: both sides derive from it, only the bytes change.
    monomial_commit: the committed matrix's basis. False = each level encodes
        `mle_evals_to_coeffs(matrix)`, so a codeword coordinate is a clean
        `eval_mle` of the eval-basis witness and the proximity glue expands
        eval-points with `eq`. True = each level encodes the bit-reversed raw
        matrix (both axes), the coefficient-basis convention of wire formats
        that commit the witness's raw lanes (flock's `ligero_commit`): the
        codeword coordinate becomes the monomial-basis evaluation
        `<slice, expand_monomial(reversed(eval_point(s)))>`, the proximity glue
        expands monomially, and the verifier's lane weights bit-reverse (eq of
        the reversed fold challenges). Prover and verifier both derive from it;
        the recursion's algebra is the same up to the basis change, only the
        committed and glued bytes differ.
    """

    num_vars: int
    fold_ks: tuple[int, ...]
    log_inv_rates: tuple[int, ...]
    queries: tuple[int, ...]
    ood_samples: tuple[int, ...] = ()
    alpha_lsb_first: bool = False
    compressed_sumcheck_messages: bool = False
    monomial_commit: bool = False

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
        if self.ood_samples:
            if len(self.ood_samples) != self.num_levels - 1:
                raise ValueError(
                    "ood_samples binds one entry per recursive commit "
                    f"(M_1..M_{self.num_levels - 1}); expected "
                    f"{self.num_levels - 1} entries, got {len(self.ood_samples)}"
                )
            if any(n < 0 for n in self.ood_samples):
                raise ValueError(
                    f"ood_samples must be non-negative; got {self.ood_samples}"
                )

    @property
    def num_levels(self) -> int:
        return len(self.fold_ks)

    @property
    def total_ood(self) -> int:
        """OOD eval-claims across the whole open (`ood_values`'s wire length)."""
        return sum(self.ood_samples)

    def ood_count(self, level: int) -> int:
        """OOD blocks run after level `level`'s recursive commit (0 when unset
        or final — the final level commits nothing)."""
        if not self.ood_samples or level >= self.num_levels - 1:
            return 0
        return self.ood_samples[level]

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
        "pow_witnesses",
        "component_positions",
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
    ood_values: the claimed out-of-domain evaluations, one per OOD block in
        schedule order (empty when `ood_samples` is `()`). Claimed, not proven
        here — each is glued into the running sumcheck, which enforces it.
    pow_witnesses: proof-of-work witnesses, one per grind the choreography
        schedules (`LigeritoChoreography.fold_grind_bits` / `query_grind_bits`), in
        schedule order (empty under the default no-grinding choreography).
    component_positions: the sampled query positions of each `component_openings`
        entry (one sorted int32 array per committed level, ascending distinct).
        The verifier re-derives these from the transcript, so they are redundant
        for verification — carried so a consumer can serialize a deduplicated
        multi-proof (e.g. flock's octopus) whose layout is positional and thus
        not recoverable from the per-query `Opening.path` alone.
    """

    sumcheck_messages: list[Array]
    recursive_roots: list[Array]
    component_openings: list[Opening]
    final_residual: Array
    ood_values: list[Array] = field(default_factory=list)
    pow_witnesses: list[Array] = field(default_factory=list)
    component_positions: list[Array] = field(default_factory=list)
