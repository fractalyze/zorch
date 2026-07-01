# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device sibling of `byte_transcript.Sha256Transcript`.

Same Merlin-over-SHA-256 byte framing (op tags, u64-LE length prefixes,
`SHA256(buffer ‖ ctr)` counter-squeeze, re-absorb of the squeezed bytes), but the
SHA-256 compression runs ON DEVICE via the name-routed `zorch.sha256` marker
(`zorch.hash.sha256.digest`) instead of host `hashlib`. This is the first step of
moving flock's Fiat-Shamir on-device (fractalyze/flock-zorch#6): the marker lowers
the byte-hash chain to a GPU kernel, unlike the host `Sha256Transcript`, so this
transcript reports `has_dedicated_fusion = True` (the device-byte row of the
`docs/transcript.md` taxonomy).

Byte-identical to the host transcript by construction — the absorbed-byte stream
is built with the identical framing, and `zorch.hash.sha256.digest` is
byte-identical to `hashlib.sha256` (pinned by `hash/testing/sha256_test.py`).

Slice status (#6): this keeps the host's growing-`bytes` buffer and re-hashes it
per squeeze, so the transcript is *device-hashed* but not yet a fixed-shape
`lax.scan` carry. A later slice replaces the buffer with a streaming
Merkle–Damgård midstate (`h[8]` + `<64 B` pending + length) so the state threads
`@jit`/`lax.scan`, and adds the `Transcript`/`GrindingTranscript` field-element
surface + PoW grind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zorch.byte_transcript import (
    KIND_SCALAR,
    KIND_SLICE,
    OP_BYTES,
    OP_DOMAIN,
    OP_LABEL,
    OP_OBSERVE,
    OP_SQUEEZE,
)
from zorch.hash import sha256 as device_sha256


def _len8(n: int) -> bytes:
    """A length / nonce as 8 little-endian bytes — the transcript's only integer
    encoding (fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


def _device_digest(buffer: bytes) -> bytes:
    """`SHA256(buffer)` via the device `zorch.sha256` marker (no host hashlib)."""
    msg = np.frombuffer(buffer, dtype=np.uint8).reshape(1, -1)
    return bytes(np.asarray(device_sha256.digest(msg)[0]))


def _device_squeeze(buffer: bytes, n: int) -> bytes:
    """`n` pseudorandom bytes as `SHA256(buffer ‖ ctr_le8)` for ctr=0,1,… (32 B per
    block, SHA-256 is not an XOF), computed on device. The counter blocks share a
    length, so the whole squeeze is ONE batched marker call (`digest` over a
    `[nblocks, len(buffer)+8]` batch) — the data-parallel use the marker is for."""
    if n <= 0:
        return b""
    nblocks = (n + 31) // 32
    msgs = np.stack(
        [np.frombuffer(buffer + _len8(ctr), dtype=np.uint8) for ctr in range(nblocks)]
    )  # [nblocks, len(buffer)+8]
    digs = np.asarray(device_sha256.digest(msgs))  # [nblocks, 32] uint8
    return digs.reshape(-1).tobytes()[:n]


@dataclass(frozen=True)
class DeviceSha256Transcript:
    """Merlin-style byte duplex over the device SHA-256 marker. Functional: every op
    returns a new transcript whose `buffer` is the running absorbed-byte stream —
    identical framing to `byte_transcript.Sha256Transcript`, hashed on device."""

    buffer: bytes

    @property
    def has_dedicated_fusion(self) -> bool:
        # The SHA-256 chain lowers to a GPU kernel via the zorch.sha256 marker —
        # the device-byte transcript's defining difference from the host one.
        return True

    @classmethod
    def new(cls, domain: bytes) -> DeviceSha256Transcript:
        """Seed with a length-prefixed domain so prefix domains can't collide:
        `[OP_DOMAIN] || len8(domain) || domain`."""
        return cls(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))

    # ---- internal absorb / squeeze ----
    def _absorb(self, payload: bytes) -> DeviceSha256Transcript:
        return DeviceSha256Transcript(self.buffer + payload)

    def _digest(self) -> bytes:
        """`SHA256(buffer)` — the proof-of-work state digest (no counter, no tag)."""
        return _device_digest(self.buffer)

    def _squeeze(self, n: int) -> bytes:
        return _device_squeeze(self.buffer, n)

    # ---- observe (byte-identical framing to the host transcript) ----
    def observe_label(self, label: bytes) -> DeviceSha256Transcript:
        return self._absorb(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))

    def observe_bytes(self, data: bytes) -> DeviceSha256Transcript:
        return self._absorb(bytes([OP_BYTES]) + _len8(len(data)) + bytes(data))

    def observe_scalar(self, payload: bytes) -> DeviceSha256Transcript:
        # No length prefix — a scalar's width is implicit in the consumer.
        return self._absorb(bytes([OP_OBSERVE, KIND_SCALAR]) + bytes(payload))

    def observe_slice(self, payload: bytes, count: int) -> DeviceSha256Transcript:
        return self._absorb(
            bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count) + bytes(payload)
        )

    # ---- sample (absorb tag, squeeze without mutating, re-absorb the squeeze) ----
    def sample_scalar(self, nbytes: int) -> tuple[DeviceSha256Transcript, bytes]:
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SCALAR]))
        buf = t._squeeze(nbytes)
        return t._absorb(buf), buf

    def sample_slice(
        self, count: int, width: int
    ) -> tuple[DeviceSha256Transcript, bytes]:
        t = self._absorb(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(count))
        buf = t._squeeze(count * width)
        return t._absorb(buf), buf
