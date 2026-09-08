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

import math
from typing import TYPE_CHECKING

import numpy as np
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes

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


# The byte sampler builds its draws from little-endian u64 chunks, so a
# ring whose `q` does not fit one cannot be drawn against. A property of
# `lattice_frx.sampler`, not of any one protocol — which is why the ceiling
# and its message live here rather than once per layer that squeezes.
_MAX_MODULUS = 1 << 64


def sampling_modulus(ring: "HostSplitRing") -> int:
    """`q` as one integer, gated for the uniform sampler's u64 chunking.

    Every layer that squeezes `Z_q` draws off the transcript needs this
    number and this gate; two layers that each derived them raised two
    near-identical errors, and the ceiling would have had to be found twice
    the day the sampler widens."""
    modulus = math.prod(ring.q_moduli)
    if modulus >= _MAX_MODULUS:
        raise ValueError(
            f"transcript: uniform draws come off u64 chunks, so q must be "
            f"below 2^64; this ring's q has {modulus.bit_length()} bits. An "
            f"RNS chain this wide needs a wider sampler first."
        )
    return modulus


def squeeze_uniform(
    transcript: ByteTranscript, label: bytes, modulus: int, count: int
) -> tuple[ByteTranscript, np.ndarray]:
    """Absorb `label`, then squeeze `count` uniform `Z_q` draws.

    The squeeze half of this module's charter. `uniform_bytes_needed` and
    `uniform_from_bytes` must be called at the *same* modulus and count or
    the byte stream desynchronises, and a protocol that respelled the
    pairing would fork the wire exactly as one respelling `absorb_stacks`
    would — with the same invisibility, since both sides of that protocol
    would still agree with each other.

    Reshaping the draws, and what to build from them, stays with the caller:
    that part genuinely differs (a ring element per relation for Fig. 7, a
    `λ×M` scalar matrix for the ENS20 aggregation).

    `range.py` deliberately does *not* come through here — its `Bin_1`
    matrix has a power-of-two support, where `uniform_from_bytes`' general
    path would spend a full u64 per two-bit draw. That is an exception to
    this rule, and it reads as one only because the rule is named.
    """
    t = transcript.observe_label(label)
    t, raw = t.sample_scalar(uniform_bytes_needed(modulus, count))
    return t, uniform_from_bytes(raw, modulus, count)
