# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold proof type. Co-located with the scheme like `FriProof` in
`fri/config.py` (kzg needs no proof dataclass) — there is no `proof.py` in
`pcs/`, so this keeps the three schemes consistent. The staggered batch weights
both `open` and `verify` derive live in `pcs/basefold/batching.py`."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeAlias

import frx
from frx import Array

from zorch.commit.merkle import Opening

# A single Merkle root binding the whole column batch (the matrix commitment).
BasefoldCommitment: TypeAlias = Array


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "univariate_messages",
        "fri_roots",
        "final_poly",
        "component_openings",
        "query_openings",
        "pow_witnesses",
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
    pow_witnesses: proof-of-work witnesses, one per grind the choreography
        schedules (`BasefoldChoreography.fold_grind_bits` / `query_grind_bits`),
        in schedule order — the wire slot mirroring `LigeritoProof.pow_witnesses`.
        Empty on the native wire, which grinds nothing; a scheduled grind is
        currently refused by the prover (this freezes the proof shape, but the
        grind production + verifier `check_grind` are a deferred delta — see the
        grind guards in `prover.py` / `verifier.py`).
    """

    univariate_messages: list[tuple[Array, Array]]
    fri_roots: list[Array]
    final_poly: Array
    component_openings: list[Opening]
    query_openings: list[Opening]
    pow_witnesses: list[Array] = field(default_factory=list)


@dataclass(frozen=True)
class CadenceProof:
    """The generic artifacts a non-native (row-batch-prefix + multi-arity-epoch)
    open produces, for a consumer to assemble into its own wire format.

    The fold schedule commits several layers with different leaf groupings, so a
    single `BasefoldProof` (uniform per-round pair leaves) does not describe it;
    this carries the raw per-layer openings + the interleaved-sumcheck artifacts
    for the consumer to serialize.

    round_messages: per fold round, the kernel's raw message-component tuple
        (e.g. `(u0, u2)`), in order.
    commit_roots: the commit roots the choreography observed, in order — the
        post-prefix root first, then one per committed epoch (all but the last).
    final_codeword: the fully folded codeword, cleartext.
    final_state: the kernel's terminal value(s) (`kernel.final`).
    layer_openings / layer_positions / layer_num_leaves: one entry per committed
        layer at the sampled query positions — layer 0 the initial codeword
        (leaf index = the full position), layer 1 the post-prefix commit, layers
        2+ the per-epoch commits (leaf index = position >> the layer's shift).
    positions: the sampled query positions (full index).
    pow_witnesses: proof-of-work witnesses, one per grind the choreography
        schedules, in schedule order — the wire slot symmetric to
        `BasefoldProof.pow_witnesses`. Empty today; the cadence open refuses a
        scheduled grind (the grind production is a deferred delta), so this
        freezes the artifact shape without wiring the grind.
    """

    round_messages: list[tuple]
    commit_roots: list[Array]
    final_codeword: Array
    final_state: Any
    layer_openings: list[Opening]
    layer_positions: list[Array]
    layer_num_leaves: list[int]
    positions: Array
    pow_witnesses: list[Array] = field(default_factory=list)


@dataclass(frozen=True)
class BasefoldConfig:
    """BaseFold fold-schedule knobs.

    num_vars: number of folded variables.
    num_queries: number of query positions opened per commitment.
    row_batch_prefix, fold_arities: fold schedule. Rounds fold arity-2 each; a
        "commit boundary" observes a root. `row_batch_prefix` rounds fold the
        codeword by a deferred multilinear lane-combine (not FRI) and commit
        once at the prefix end; the remaining rounds fold per-round and commit
        at the cumulative epoch boundaries.
    """

    num_vars: int
    num_queries: int
    row_batch_prefix: int = 0  # 0 = no prefix (zorch-native)
    fold_arities: tuple[int, ...] = ()  # () = uniform per-round commit (zorch-native)

    def __post_init__(self) -> None:
        # A non-native cadence schedule must cover exactly `num_vars` rounds:
        # `row_batch_prefix` (prefix folds) + `sum(fold_arities)` (FRI folds). A
        # mismatch would overrun `fold_arities` (IndexError) or leave later epochs
        # uncommitted, so fail fast at construction. The native schedule is
        # implicit (per-round), so it is exempt.
        if not self.commits_per_round:
            scheduled = self.row_batch_prefix + sum(self.fold_arities)
            if scheduled != self.num_vars:
                raise ValueError(
                    f"cadence fold schedule covers {scheduled} rounds "
                    f"(row_batch_prefix={self.row_batch_prefix} + "
                    f"sum(fold_arities)={sum(self.fold_arities)}) but "
                    f"num_vars={self.num_vars}"
                )

    @property
    def commits_per_round(self) -> bool:
        return not self.fold_arities and self.row_batch_prefix == 0

    def require_native(self, action: str) -> None:
        """Fail loud on a non-native fold schedule: the native drivers handle only
        `commits_per_round` (pre-fold arity-2 pair commit every round). The
        row-batch-prefix + multi-arity epoch cadence is the deferred fold-schedule
        machinery (design §"Core driver"), wired + byte-gated with its first
        byte-fixed consumer; only the config STRUCTURE for it exists here.
        `action` names the caller's verb (e.g. "open" / "verify") for the message,
        so prover and verifier share one guard."""
        if not self.commits_per_round:
            raise NotImplementedError(
                f"non-native fold cadence (row_batch_prefix / fold_arities) is not "
                f"wired to {action} yet; only commits_per_round (zorch-native) is "
                f"wired. The deferred row-batch-prefix + multi-arity epoch cadence "
                f"lands with its first byte-fixed consumer"
            )

    def layer_shifts(self) -> list[int]:
        """The per-committed-layer index shift, in commit order: layer 0 the
        initial commit at the full query index, then one shift per committed layer
        (the post-prefix commit, then each committed epoch), each the cumulative
        fold count above that layer. A query position maps to a layer's leaf index
        by `position >> shift`. Pure function of `fold_arities` — the running fold
        total at each commit boundary — so the prover's interleaved commits and the
        verifier's replay read the identical sequence off one method."""
        return list(itertools.accumulate(self.fold_arities, initial=0))
