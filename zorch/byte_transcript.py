# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-oriented Fiat-Shamir transcript: the `ByteTranscript` seam and a
Merlin-over-hash duplex (`ByteHashTranscript`) parameterized by a `ByteHash`.

This is the HOST-side, byte-oriented sibling of the device-resident algebraic
`DuplexTranscript` (`transcript.py`). The two form a taxonomy (see
`docs/blocks/transcript.md`): an algebraic sponge whose `observe`/`sample` are *device*
ops fused into the round body, vs a byte hash whose Fiat-Shamir chain is strictly
sequential and runs on the host.

`ByteHashTranscript` holds a `bytes` buffer and an injected `ByteHash`
(`hash/byte_hash.py`); the digest substrate — host `HostSha256` or the device
`Sha256` marker — is a value it carries, not a class it hardcodes. Its
`has_dedicated_fusion` delegates to that hash, exactly as `DuplexSponge` delegates
to its `Permutation`. So the same construction backs both a host byte challenger
(inject `HostSha256()`) and the device-byte row of the taxonomy (inject
`Sha256()`); the two are byte-identical.

The construction — op-tagged absorb, `HASH(buffer || ctr)` counter-squeeze (SHA-256
is not an XOF), and re-absorb of the squeezed bytes — is a standard Merlin-style
duplex, byte-identical to the canonical SHA-256 Fiat-Shamir transcript used by
binary-field provers (e.g. succinctlabs/flock's `FsChallenger`). zorch owns the
wire framing on OPAQUE bytes; a consumer supplies only its field<->bytes
serialization (see flock-zorch's `challenger.py`, an F128 (16-byte lo||hi) surface).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np

if TYPE_CHECKING:
    from zorch.hash.byte_hash import ByteHash

# Wire vocabulary — the Merlin-over-hash framing (op byte, then operands).
OP_DOMAIN = 0x01
OP_LABEL = 0x02
OP_OBSERVE = 0x03
OP_SQUEEZE = 0x04
OP_BYTES = 0x05
KIND_SCALAR = 0x01
KIND_SLICE = 0x02

# Proof-of-work: nonces a device batch tests per `digest` call. A `bits`-hit lands
# in the first window for any practical `bits`, so a fusion hash normally grinds in
# one marker call; a host hash uses window 1 (sequential early-exit, like hashlib).
_GRIND_WINDOW = 1 << 16


def _len8(n: int) -> bytes:
    """A length / nonce as 8 little-endian bytes (the transcript's only integer
    encoding — fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


def _leading_zero_bits_ok(digests: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized PoW predicate: whether each digest (uint8 `[B, digest_size]`) has
    >= `bits` leading zero bits, big-endian (digest[0] most significant)."""
    full, extra = divmod(bits, 8)
    ok = np.all(digests[:, :full] == 0, axis=1)
    if extra:
        ok &= (digests[:, full] >> np.uint8(8 - extra)) == 0
    return ok


def _validate_pow_bits(bits: int, digest_size: int) -> None:
    """Reject a PoW target outside `[0, digest_size * 8]`: a negative count, or
    more leading-zero bits than the digest carries (e.g. 264 on a 256-bit digest,
    which `_leading_zero_bits_ok` would spuriously accept for an all-zero digest).
    The byte sibling of `transcript._validate_pow_bits`."""
    if not 0 <= bits <= digest_size * 8:
        raise ValueError(f"bits must be in [0, {digest_size * 8}], got {bits}")


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
    def grind_pow(self, bits: int) -> tuple[Self, int]: ...
    def verify_pow(self, nonce: int, *, bits: int) -> tuple[Self, bool]: ...


@dataclass(frozen=True)
class ByteHashTranscript:
    """Merlin-style byte duplex over an injected `ByteHash`. Functional: every op
    returns a new transcript whose `buffer` is the running absorbed-byte stream. A
    host object (a `bytes` buffer, not a jit-traced pytree); the `ByteHash` chooses
    the squeeze substrate — `HostSha256` (host `hashlib`) or `Sha256` (the
    `zorch.sha256` device marker). Byte-identical whichever is injected."""

    buffer: bytes
    byte_hash: ByteHash

    @property
    def has_dedicated_fusion(self) -> bool:
        # Delegates to the hash — as `DuplexSponge.has_dedicated_fusion` delegates
        # to its `Permutation`. Names no concrete hash.
        return self.byte_hash.has_dedicated_fusion

    @classmethod
    def new(cls, domain: bytes, byte_hash: ByteHash) -> ByteHashTranscript:
        """Seed with a length-prefixed domain so prefix domains can't collide:
        `[OP_DOMAIN] || len8(domain) || domain`."""
        return cls(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain), byte_hash)

    # ---- internal absorb / squeeze (over byte_hash.digest) ----
    def _absorb(self, payload: bytes) -> ByteHashTranscript:
        return ByteHashTranscript(self.buffer + payload, self.byte_hash)

    def _digest(self) -> bytes:
        """`HASH(buffer)` — the proof-of-work state digest (no counter, no tag)."""
        row = np.frombuffer(self.buffer, dtype=np.uint8)[None, :]  # [1, len(buffer)]
        return bytes(np.asarray(self.byte_hash.digest(row))[0])

    def _squeeze(self, n: int) -> bytes:
        """`n` bytes as `HASH(buffer || ctr_le8)` for ctr=0,1,… (digest_size B per
        block, a hash is not an XOF). The counter blocks share a length, so the
        whole squeeze is ONE batched `byte_hash.digest` call."""
        if n <= 0:
            return b""
        d = self.byte_hash.digest_size
        nblocks = (n + d - 1) // d
        batch = np.stack(
            [
                np.frombuffer(self.buffer + _len8(ctr), dtype=np.uint8)
                for ctr in range(nblocks)
            ]
        )  # [nblocks, len(buffer)+8]
        digs = np.asarray(self.byte_hash.digest(batch))  # [nblocks, digest_size]
        return digs.reshape(-1).tobytes()[:n]

    # ---- observe ----
    def observe_label(self, label: bytes) -> ByteHashTranscript:
        return self._absorb(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))

    def observe_bytes(self, data: bytes) -> ByteHashTranscript:
        return self._absorb(bytes([OP_BYTES]) + _len8(len(data)) + bytes(data))

    def observe_scalar(self, payload: bytes) -> ByteHashTranscript:
        # No length prefix — a scalar's width is implicit in the consumer.
        return self._absorb(bytes([OP_OBSERVE, KIND_SCALAR]) + bytes(payload))

    def observe_slice(self, payload: bytes, count: int) -> ByteHashTranscript:
        return self._absorb(
            bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count) + bytes(payload)
        )

    # ---- sample (absorb tag, squeeze without mutating, re-absorb the squeeze) ----
    def sample_scalar(self, nbytes: int) -> tuple[ByteHashTranscript, bytes]:
        if nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SCALAR]))
        buf = t._squeeze(nbytes)
        return t._absorb(buf), buf

    def sample_slice(self, count: int, width: int) -> tuple[ByteHashTranscript, bytes]:
        if count < 0 or width < 0:
            raise ValueError(f"count/width must be non-negative, got {count}/{width}")
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(count))
        buf = t._squeeze(count * width)
        return t._absorb(buf), buf

    # ---- proof-of-work ----
    def _grind(self, state_digest: bytes, bits: int) -> int:
        """Lowest u64 nonce with `HASH(state_digest || nonce_le8)` having `bits`
        leading zero bits. Tests a window of nonces per `digest` call (window 1 for
        a host hash = sequential early-exit); tiles windows until a hit — unbounded,
        never returns an unchecked nonce."""
        window = _GRIND_WINDOW if self.byte_hash.has_dedicated_fusion else 1
        base = 0
        while True:
            batch = np.stack(
                [
                    np.frombuffer(state_digest + _len8(base + i), dtype=np.uint8)
                    for i in range(window)
                ]
            )  # [window, len(state_digest)+8]
            hits = np.flatnonzero(
                _leading_zero_bits_ok(np.asarray(self.byte_hash.digest(batch)), bits)
            )
            if hits.size:
                return base + int(hits[0])
            base += window

    def grind_pow(self, bits: int) -> tuple[ByteHashTranscript, int]:
        """Lowest u64 nonce whose PoW passes (0 if bits==0), then absorb it via
        `observe_bytes` so subsequent challenges bind to it."""
        _validate_pow_bits(bits, self.byte_hash.digest_size)
        nonce = 0 if bits == 0 else self._grind(self._digest(), bits)
        return self.observe_bytes(_len8(nonce)), nonce

    def verify_pow(self, nonce: int, *, bits: int) -> tuple[ByteHashTranscript, bool]:
        """Verifier mirror: check the PoW (bits==0 requires the canonical nonce 0),
        then absorb the nonce REGARDLESS so the transcript stays in lockstep."""
        _validate_pow_bits(bits, self.byte_hash.digest_size)
        if bits == 0:
            ok = nonce == 0
        else:
            row = np.frombuffer(self._digest() + _len8(nonce), dtype=np.uint8)[None, :]
            ok = bool(
                _leading_zero_bits_ok(np.asarray(self.byte_hash.digest(row)), bits)[0]
            )
        return self.observe_bytes(_len8(nonce)), ok


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bt: type[ByteTranscript] = ByteHashTranscript
