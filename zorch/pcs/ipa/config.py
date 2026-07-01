# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""IPA wire types on the `pcs` seam.

The commitment is a bare array — a batch of G1 points, one Pedersen
commitment `P = ⟨a, G⟩` per polynomial — so it is a named alias like KZG's. The
proof carries structure (the per-round cross terms plus the collapsed scalar) and
crosses the open/verify `@jit` boundary, so it is a registered-pytree dataclass
like `FriProof`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import jax
from jax import Array

IpaCommitment: TypeAlias = Array  # G1 affine [K] — one P = ⟨a, G⟩ per poly


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["l", "r", "a"],
    meta_fields=[],
)
@dataclass(frozen=True)
class IpaProof:
    """One polynomial's opening proof.

    `l`, `r` are the round cross terms `L_j`, `R_j` (G1 affine, `[k]` each
    for `k = log₂ n` rounds); `a` is the single field scalar the coefficient
    vector collapses to after the last fold. The verifier needs no folded `b`
    scalar in the proof — it recomputes `b = h(x)` from the round challenges in
    O(log n) (see `math.eval_challenge_poly`). A registered pytree so the proof
    crosses the `open`/`verify` `@jit` boundary."""

    l: Array  # G1 affine [k] — L_j per round
    r: Array  # G1 affine [k] — R_j per round
    a: Array  # scalar field — collapsed coefficient after k folds


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["l", "r", "a", "hiding_comm", "rand"],
    meta_fields=[],
)
@dataclass(frozen=True)
class IpaZkProof:
    """One polynomial's *hiding* opening proof — the no-zk `IpaProof` plus the two
    elements blinding adds.

    `l`, `r`, `a` are the fold's cross terms and collapsed scalar exactly as in
    `IpaProof`: the zk fold runs the *same* recurrence, only on a blinded
    `(commitment, coeffs)`. `hiding_comm` is the Pedersen commitment to the blinding
    polynomial (`⟨hiding_poly, G⟩ + s·hiding_rand`) and `rand` the accumulated
    commitment randomness; the verifier re-folds both into the statement before
    replaying the fold (see `prover._open_one_zk` / `verifier.reduce_opening_zk`). A
    registered pytree so it crosses the `open`/`verify` `@jit` boundary."""

    l: Array  # G1 affine [k] — L_j per round
    r: Array  # G1 affine [k] — R_j per round
    a: Array  # scalar field — collapsed coefficient after k folds
    hiding_comm: Array  # G1 affine — Pedersen commitment of the blinding poly
    rand: Array  # scalar field — accumulated commitment randomness
