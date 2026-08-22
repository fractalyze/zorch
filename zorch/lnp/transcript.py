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

A protocol module that respelled either half would fork the wire, and no
single-module test can see it — both sides of *that* protocol would agree
with each other while disagreeing with its sibling. Hence one definition,
imported.
"""

from __future__ import annotations

import numpy as np

from zorch.byte_transcript import ByteTranscript


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
