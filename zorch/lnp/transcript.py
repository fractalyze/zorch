# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The lnp transcript wire: how a ring stack becomes transcript bytes.

Every protocol in this package absorbs the same kind of object — stacks of
ring elements in canonical residue form — and the two sides of a
Fiat-Shamir proof must agree on the encoding down to the byte. That
agreement is two decisions, and they belong together in one place:

- **Serialization.** Canonical residues as C-order little-endian u64. The
  ring's public contract is already `uint64` arrays, so this is a
  reinterpretation and a copy, never a conversion.
- **The slice count.** `observe_slice` is told how many *ring elements* a
  payload carries, not how many machine words. The count is domain-separation
  input, so the two conventions produce different transcripts from the same
  bytes; picking the element count keeps the transcript describing the
  algebra rather than the storage layout.

The squeeze direction is the same decision seen from the other side: a
layer that aggregates N statements draws its weights off the transcript,
and prover and verifier must derive them from the same bytes in the same
order. `Π_eval`'s `γ ∈ Z_q` and `Π_many^(2)`'s `µ ∈ R_q` differ only in
which domain they land in, so the byte derivation is stated once here and
each layer states only its domain.

A protocol module that respelled any of it would fork the wire, and no
single-module test can see it — both sides of *that* protocol would agree
with each other while disagreeing with its sibling. Hence one definition,
imported.
"""

from __future__ import annotations

import math

import numpy as np
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript

# The byte sampler builds its draws from little-endian u64 chunks, so a
# modulus at or above this has no uniform draw on this wire.
_MAX_MODULUS = 1 << 64


def stack_bytes(stack: np.ndarray) -> bytes:
    """A ring stack's canonical residues as C-order little-endian u64.

    The dtype gate is what makes this a reinterpretation rather than a
    conversion, which is the claim the module docstring makes above. Without
    it `astype` would quietly *reinterpret* a signed array — `-1` and the
    residue `2**64 - 1` produce byte-identical transcript input — and
    truncate a float one. Two different values binding to one transcript is
    the failure this wire exists to prevent, so it is refused at the door
    rather than trusted to every caller.

    It gates dtype only, deliberately: injectivity of the byte map is a
    dtype property, and an out-of-range residue creates no collision. That
    is also all this function *can* check — it never sees the moduli. So
    the error says "uint64", not "canonical", which in lattice-frx's
    vocabulary would also promise every residue is below its modulus."""
    if stack.dtype != np.uint64:
        raise TypeError(
            f"stack_bytes: expected native uint64 residues, got dtype "
            f"{stack.dtype!r} — a signed or float array would reinterpret "
            f"or truncate rather than serialize"
        )
    return np.ascontiguousarray(stack).astype("<u8", copy=False).tobytes()


def absorb_stacks(transcript: ByteTranscript, *stacks: np.ndarray) -> ByteTranscript:
    """Absorb ring stacks in order, each counted in ring elements.

    `ByteTranscript` is functional, so this returns the advanced transcript
    and leaves the caller's value untouched — which is what lets a rejected
    Fiat-Shamir-with-aborts attempt restart from the original."""
    for stack in stacks:
        transcript = transcript.observe_slice(stack_bytes(stack), stack.shape[0])
    return transcript


def require_u64_modulus(ring: HostSplitRing, who: str) -> int:
    """`q = Π q_i` as one Python int, gated to what `squeeze_uniform` can
    draw from.

    Every layer that squeezes a transcript-derived weight needs this bound,
    and it is the sampler's, not any one protocol's — so it is checked at
    construction, where the caller can still choose a different ring, and
    named after the layer that asked."""
    modulus = math.prod(ring.q_moduli)
    if modulus >= _MAX_MODULUS:
        raise ValueError(
            f"{who}: transcript draws are built from little-endian u64 "
            f"chunks, so q must be below 2^64; this ring's q has "
            f"{modulus.bit_length()} bits. An RNS chain this wide needs a "
            f"wider sampler first."
        )
    return modulus


def squeeze_uniform(
    transcript: ByteTranscript, modulus: int, count: int
) -> tuple[ByteTranscript, np.ndarray]:
    """Squeeze `count` uniform residues mod `modulus`, and the advanced
    transcript.

    The byte count is derived from the modulus rather than passed, which is
    what keeps two layers squeezing the same number of bytes for the same
    draw — the squeeze-side twin of `absorb_stacks`. What the residues then
    *mean* is the caller's: `Π_eval` reads them as `Z_q` scalars, Fig. 7 as
    the coefficients of `R_q` elements."""
    transcript, raw = transcript.sample_scalar(uniform_bytes_needed(modulus, count))
    return transcript, uniform_from_bytes(raw, modulus, count)
