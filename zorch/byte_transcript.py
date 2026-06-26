# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-oriented Fiat-Shamir transcript: the `ByteTranscript` seam and a
Merlin-over-SHA-256 duplex (`Sha256Transcript`).

This is the HOST-side, byte-oriented sibling of the device-resident algebraic
`DuplexTranscript` (`transcript.py`). The two form a taxonomy (see
`docs/transcript.md`): an algebraic sponge whose `observe`/`sample` are *device*
ops fused into the round body, vs a byte hash whose Fiat-Shamir chain is strictly
sequential and runs on the host. `Sha256Transcript` reports
`has_dedicated_fusion = False` — the type-level signal that it is a host
orchestrator, not a device kernel, and is therefore exempt from the
device-fusion clause (a SHA-256 byte chain does not lower to a GPU kernel; the
bulk arithmetic *between* challenges still fuses per-island).

The construction — op-tagged absorb, `SHA256(buffer || ctr)` counter-squeeze
(SHA-256 is not an XOF), and re-absorb of the squeezed bytes — is a standard
Merlin-style duplex. It is byte-identical to the canonical SHA-256 Fiat-Shamir
transcript used by binary-field provers (e.g. succinctlabs/flock's
`FsChallenger`). zorch owns the wire framing on OPAQUE bytes; a consumer supplies
only its field<->bytes serialization (the element width) — see flock-zorch's
`challenger.py`, which wraps this with an F128 (16-byte lo||hi) surface.

Squeeze backend: host `hashlib.sha256` — the FIPS-180-4 reference, with no
per-length recompile (the buffer grows every op, which would re-trace a jitted
hash each call). The data-parallel device sibling `zorch.hash.sha256.digest` is
byte-identical (pinned by `hash/testing/sha256_test.py`) and is for batched
Merkle / proof-of-work grinding, not this sequential transcript.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

# Wire vocabulary — the Merlin-over-SHA256 framing (op byte, then operands).
OP_DOMAIN = 0x01
OP_LABEL = 0x02
OP_OBSERVE = 0x03
OP_SQUEEZE = 0x04
OP_BYTES = 0x05
KIND_SCALAR = 0x01
KIND_SLICE = 0x02


def _len8(n: int) -> bytes:
    """A length / nonce as 8 little-endian bytes (the transcript's only integer
    encoding — fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _has_leading_zero_bits(h: bytes, bits: int) -> bool:
    """Whether digest `h` has at least `bits` leading zero bits, big-endian
    (h[0] most significant) — the proof-of-work predicate."""
    full, extra = divmod(bits, 8)
    if any(h[i] != 0 for i in range(full)):
        return False
    if extra and (h[full] >> (8 - extra)) != 0:
        return False
    return True


class ByteTranscript(Protocol):
    """Byte-oriented Fiat-Shamir seam. `observe_*` append tagged, length-prefixed
    bytes; `sample_*` squeeze raw bytes (the consumer reinterprets to field
    elements). Distinct from `transcript.Transcript`, which is field-element- and
    device-oriented."""

    def observe_label(self, label: bytes) -> Self: ...
    def observe_bytes(self, data: bytes) -> Self: ...
    def observe_scalar(self, payload: bytes) -> Self: ...
    def observe_slice(self, payload: bytes, count: int) -> Self: ...
    def sample_scalar(self, nbytes: int) -> tuple[Self, bytes]: ...
    def sample_slice(self, count: int, width: int) -> tuple[Self, bytes]: ...


class ByteGrindingTranscript(ByteTranscript, Protocol):
    """A `ByteTranscript` with a u64-nonce proof-of-work grind. Split from the
    base seam (as `transcript.GrindingTranscript` is) — the byte/nonce PoW is a
    different predicate from the field-element one and must not be cross-used."""

    def grind_pow(self, bits: int) -> tuple[Self, int]: ...
    def verify_pow(self, nonce: int, bits: int) -> tuple[Self, bool]: ...


@dataclass(frozen=True)
class Sha256Transcript:
    """Merlin-style byte duplex over SHA-256. Functional: every op returns a new
    transcript whose `buffer` is the running absorbed-byte stream. Forking (build
    two from the same prefix, then diverge) is plain value reuse — no aliasing."""

    buffer: bytes

    @property
    def has_dedicated_fusion(self) -> bool:
        return False

    @classmethod
    def new(cls, domain: bytes) -> Sha256Transcript:
        """Seed with a length-prefixed domain so prefix domains can't collide:
        `[OP_DOMAIN] || len8(domain) || domain`."""
        return cls(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))

    # ---- internal absorb / squeeze ----
    def _absorb(self, payload: bytes) -> Sha256Transcript:
        return Sha256Transcript(self.buffer + payload)

    def _digest(self) -> bytes:
        """`SHA256(buffer)` — the proof-of-work state digest (no counter, no tag)."""
        return _sha256(self.buffer)

    def _squeeze(self, n: int) -> bytes:
        """`n` pseudorandom bytes as `SHA256(buffer || ctr_le8)` for ctr=0,1,…
        (32 B/block), without mutating the buffer (SHA-256 is not an XOF)."""
        out = bytearray()
        ctr = 0
        while len(out) < n:
            out += _sha256(self.buffer + _len8(ctr))
            ctr += 1
        return bytes(out[:n])

    # ---- observe ----
    def observe_label(self, label: bytes) -> Sha256Transcript:
        return self._absorb(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))

    def observe_bytes(self, data: bytes) -> Sha256Transcript:
        return self._absorb(bytes([OP_BYTES]) + _len8(len(data)) + bytes(data))

    def observe_scalar(self, payload: bytes) -> Sha256Transcript:
        # No length prefix — a scalar's width is implicit in the consumer.
        return self._absorb(bytes([OP_OBSERVE, KIND_SCALAR]) + bytes(payload))

    def observe_slice(self, payload: bytes, count: int) -> Sha256Transcript:
        return self._absorb(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count) + bytes(payload))

    # ---- sample (absorb tag, squeeze without mutating, re-absorb the squeeze) ----
    def sample_scalar(self, nbytes: int) -> tuple[Sha256Transcript, bytes]:
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SCALAR]))
        buf = t._squeeze(nbytes)
        return t._absorb(buf), buf

    def sample_slice(self, count: int, width: int) -> tuple[Sha256Transcript, bytes]:
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(count))
        buf = t._squeeze(count * width)
        return t._absorb(buf), buf

    # ---- proof-of-work ----
    def grind_pow(self, bits: int) -> tuple[Sha256Transcript, int]:
        """Lowest u64 nonce with `SHA256(state_digest || nonce_le8)` having `bits`
        leading zero bits (0 if bits==0), then absorb the nonce via `observe_bytes`
        so subsequent challenges bind to it."""
        state_digest = self._digest()
        nonce = 0
        if bits != 0:
            while not _has_leading_zero_bits(_sha256(state_digest + _len8(nonce)), bits):
                nonce += 1
        return self.observe_bytes(_len8(nonce)), nonce

    def verify_pow(self, nonce: int, bits: int) -> tuple[Sha256Transcript, bool]:
        """Verifier mirror: check the PoW (bits==0 requires the canonical nonce 0),
        then absorb the nonce REGARDLESS so the transcript stays in lockstep."""
        state_digest = self._digest()
        ok = (nonce == 0) if bits == 0 else _has_leading_zero_bits(
            _sha256(state_digest + _len8(nonce)), bits
        )
        return self.observe_bytes(_len8(nonce)), ok


if TYPE_CHECKING:
    # Seam-conformance pins (docs/conventions.md).
    _bt: type[ByteTranscript] = Sha256Transcript
    _bg: type[ByteGrindingTranscript] = Sha256Transcript
