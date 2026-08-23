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

from typing import TYPE_CHECKING

import numpy as np

from zorch.byte_transcript import ByteTranscript

if TYPE_CHECKING:
    # Annotation only: `absorb_signed` calls a method on the ring it is
    # handed and never constructs one, so this module stays free of a
    # runtime lattice-frx dependency — it is the wire every other module
    # here imports, and widening its graph would widen theirs.
    from lattice_frx.split_ring import HostSplitRing


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


def absorb_signed(
    transcript: ByteTranscript, ring: "HostSplitRing", *arrays: np.ndarray
) -> ByteTranscript:
    """Absorb signed integer arrays, through the ring's canonical form.

    `stack_bytes` refuses a signed array at the door, and rightly: `-1` and
    the residue `2**64 - 1` are byte-identical, so serializing signed input
    directly is how two values bind to one transcript. But a protocol does
    sometimes have to absorb one — Fig. 9's revealed projection `⃗z` lives in
    unreduced ℤ, because its *norm* is the statement and a residue has none.

    So the reduction is named here rather than left to each caller. The
    choice being pinned is that a signed array is absorbed as the ring
    element it reduces to, and that lives beside `absorb_stacks` for the
    reason the module docstring gives: the second protocol to need this
    would otherwise pick its own convention (raw two's-complement? balanced?
    mod q?), and no single-module suite could see the fork — both sides of
    *that* protocol would agree with each other.

    Each array is `(k, d)` signed integers, absorbed as `k` ring elements.
    """
    return absorb_stacks(
        transcript, *(ring.from_signed_stack(array) for array in arrays)
    )
