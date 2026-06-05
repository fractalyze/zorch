# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared FRI parameters, proof types, and the Fiat-Shamir position derivation.

FRI is transparent, so the prover and verifier hold the *same* public params —
the degenerate case of the PCS prover-key/verifier-key split (no secret to keep
asymmetric). `FriParams` is that shared object. The position derivation lives here
so prover and verifier sample identical query indices from the transcript, the way
the sumcheck block shares one module-level oracle to keep the two sides in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree, Opening
from zorch.transcript import Transcript

FriCommitment: TypeAlias = Array  # stacked Merkle roots, one per committed poly


@dataclass(frozen=True)
class FriParams:
    """Public FRI configuration, identical on both sides."""

    code: ReedSolomon  # LDE; gives block_len + eval domain (incl. coset_shift)
    tree: MerkleTree  # Merkle commitment over codeword leaves
    num_rounds: int  # fold rounds; final codeword has block_len >> num_rounds entries
    num_queries: int  # query repetitions (soundness amplification)


@dataclass(frozen=True)
class LayerOpening:
    """A committed fold layer's conjugate-pair openings, batched over the queries
    (leading axis = query count) so the whole query phase is one device op."""

    lo: Opening  # leaves at a[layer]        (batched over queries)
    hi: Opening  # leaves at a[layer] + half (batched over queries)


@dataclass(frozen=True)
class FriProof:
    """value = claimed f(z); layer_roots = roots of committed fold layers
    1..num_rounds-1; final_layer = the last (cleartext, constant) codeword;
    f_lo/f_hi/layers carry the query openings, batched over the query axis."""

    value: Array
    layer_roots: list[Array]
    final_layer: Array
    f_lo: Opening  # base-codeword conjugate pair, to rebuild the quotient
    f_hi: Opening
    layers: list[LayerOpening]  # committed layers 1 .. num_rounds-1


def query_layer_indices(
    positions: Array, block_len: int, num_rounds: int
) -> list[Array]:
    """The folded query index `a_i` at each layer for a batch of `positions`:
    `a_i = q_i mod (n/2^{i+1})` with `q_0 = positions`, `q_{i+1} = a_i`, elementwise
    over the query axis. The conjugate to open at layer `i` is `a_i + n/2^{i+1}`."""
    indices = []
    q = positions
    for i in range(num_rounds):
        a = q % (block_len >> (i + 1))
        indices.append(a)
        q = a
    return indices


def sample_positions(
    transcript: Transcript, block_len: int, count: int
) -> tuple[Transcript, Array]:
    """Squeeze `count` query positions in `[0, block_len)` as one device int32
    array — no host round-trip — derived identically on both sides. Each squeezed
    field element's low limb is reduced mod `block_len`."""
    t, raw = transcript.sample(count)
    limbs = lax.bitcast_convert_type(raw, jnp.uint32).reshape(count, -1)
    return t, (limbs[:, 0] % block_len).astype(jnp.int32)
