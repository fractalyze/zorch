# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR parameters and proof type. Co-located with the scheme like `BasefoldProof`
in `basefold/config.py` — there is no shared `proof.py` in `pcs/`, so each scheme
keeps its proof beside its prover/verifier.

WHIR is a multilinear PCS that, unlike fri/basefold, does NOT fold a codeword in
place: each round folds the *MLE* by `k_whir` sumcheck sub-folds and re-encodes
the folded MLE as a fresh, shrinking Reed-Solomon codeword (commit + out-of-domain
sample), then opens the *previous* round's codeword at strided query cosets for
consistency. `code` is therefore an RS **encoder** (re-applied per round at a
halving message length), not a `FoldableCode`; `tree` is the query-strided Merkle
commitment whose `rows_per_query = 2^k_whir` coset is one opened leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import frx
from frx import Array

from zorch.commit.merkle import Opening

# A single Merkle root binding the committed codeword. The initial commitment and
# every re-encoded round share this wire type.
WhirCommitment: TypeAlias = Array


@dataclass(frozen=True)
class WhirParams:
    """The per-protocol WHIR knobs both prover and verifier read. Frozen and
    hashable so it rides a prover/verifier `@jit` static key by value (#214).

    k_whir: variables folded per WHIR round (the strided coset width is
        `2^k_whir`).
    num_queries: query repetitions per round; its length is the WHIR round count.
        Placeholder counts, not soundness-calibrated.
    mu_pow_bits: grinding work before sampling the batch-combine challenge μ. May
        be 0.
    folding_pow_bits / query_pow_bits: grinding work after each sumcheck message
        and before each query phase. May be 0 (the self-test runs grind-free).
    rate_increase: which per-round codeword domain schedule to use. The message
        always shrinks by `2^k_whir` per round (the sumcheck folds). False (default)
        keeps the rate fixed — the block length shrinks by `2^k_whir` too. True
        shrinks the block length by only `2^1` per round (`log_rs -= 1`, decoupled
        from `k_whir`), so the rate climbs every round: this is WHIR's
        rate-increasing schedule (and what openvm-stark-backend / SWIRL emit, where
        it is the source of WHIR's lower query count). The two coincide at
        `k_whir == 1`. Prover and verifier must agree."""

    k_whir: int
    num_queries: tuple[int, ...]
    mu_pow_bits: int = 0
    folding_pow_bits: int = 0
    query_pow_bits: int = 0
    rate_increase: bool = False


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "mu_pow_witness",
        "sumcheck_polys",
        "codeword_roots",
        "ood_values",
        "folding_pow_witnesses",
        "query_pow_witnesses",
        "initial_openings",
        "codeword_openings",
        "final_poly",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class WhirProof:
    """One WHIR open proof: per-round sumcheck messages, the re-encoded codeword
    commitments with their out-of-domain answers, the grinding witnesses, the
    strided query openings, and the final folded polynomial in the clear.

    mu_pow_witness: the grinding witness for the batch-combine challenge μ (a
        scalar; zero when `mu_pow_bits == 0`).
    sumcheck_polys: per sumcheck fold (`num_rounds·k_whir` total), the degree-2
        message as evaluations `(s(1), s(2))`; `s(0)` is recovered from the
        running claim.
    codeword_roots: the re-encoded codeword's Merkle root, one per non-final
        round (length `num_rounds − 1`).
    ood_values: the out-of-domain evaluation answered each non-final round
        (length `num_rounds − 1`).
    folding_pow_witnesses: grinding witness after each sumcheck message
        (`num_rounds·k_whir`).
    query_pow_witnesses: grinding witness before each round's query phase
        (`num_rounds`).
    initial_openings: round 0's strided openings of the initial committed
        codeword matrices at the round's query cosets — one `Opening` per
        commitment (`row` is `(Q, 2^k, Wᵢ)`; the verifier μ-combines the columns
        across all commitments). A single commitment is the length-1 case; later
        rounds open the single re-encoded codeword.
    codeword_openings: round `r`'s strided opening of round `r−1`'s re-encoded
        codeword, for `r` in `1..num_rounds` (length `num_rounds − 1`).
    final_poly: the last round's folded MLE coefficients, sent in the clear; the
        terminal constraint ties it to the running claim.
    """

    mu_pow_witness: Array
    sumcheck_polys: list[Array]
    codeword_roots: list[Array]
    ood_values: list[Array]
    folding_pow_witnesses: list[Array]
    query_pow_witnesses: list[Array]
    initial_openings: list[Opening]
    codeword_openings: list[Opening]
    final_poly: Array
