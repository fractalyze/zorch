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

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_dataclass

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


# ---------------------------------------------------------------------------
# Streaming Merkle–Damgård midstate (the fixed-shape, scan-threadable core).
#
# The class above keeps the whole absorbed buffer and re-hashes it per squeeze —
# device-hashed, but the growing buffer is not a valid `lax.scan` carry. This
# primitive keeps SHA-256's incremental state instead: the midstate over every
# COMPLETE 64-byte block, plus the (<64 B) trailing partial block and the running
# byte length. All shapes are fixed, so `Sha256State` threads `@jit` / a scan
# carry. A squeeze `SHA256(buffer ‖ ctr)` is `finalize(state, ctr_le8)` — a
# non-mutating copy that pads at the current length — reproducing the class's
# bytes exactly, incrementally. A later slice builds the field-element
# `Transcript` / `GrindingTranscript` surface on this so flock's Fiat-Shamir round
# loop collapses into one device program (flock-zorch#6).
# ---------------------------------------------------------------------------

_BLOCK = 64  # SHA-256 block size in bytes


@register_dataclass
@dataclass(frozen=True)
class Sha256State:
    """Incremental SHA-256 state as a JAX pytree. Fixed shapes → scan-threadable."""

    h: Array  # uint32[8] — midstate over all complete 64-byte blocks so far
    pending: Array  # uint8[64] — trailing partial block, valid prefix [:pending_len]
    pending_len: Array  # int32 — 0..63
    total_len: Array  # int32 — total bytes absorbed


def sha256_stream_init() -> Sha256State:
    """A fresh incremental hash (no bytes absorbed)."""
    return Sha256State(
        h=device_sha256.INITIAL_STATE,
        pending=jnp.zeros(_BLOCK, dtype=jnp.uint8),
        pending_len=jnp.int32(0),
        total_len=jnp.int32(0),
    )


def sha256_stream_absorb(state: Sha256State, data: Array) -> Sha256State:
    """Absorb `data` (uint8 [L], L static) into the incremental hash: fold every
    newly-complete 64-byte block into the midstate, keep the `<64 B` remainder as
    the new pending block. The block loop is a Python-unrolled, active-count-masked
    schedule over STATIC slices (never a traced-index gather / scan-carry scatter)
    — the fractalyze/zkx#500 CPU-safe pattern `transcript.DuplexTranscript` uses."""
    length = data.shape[0]
    pl = state.pending_len
    combined_src = jnp.concatenate([state.pending, data.astype(jnp.uint8)])  # [64+L]
    new_len = pl + jnp.int32(length)
    active_blocks = new_len // _BLOCK
    max_blocks = (_BLOCK - 1 + length) // _BLOCK  # static upper bound

    # Drop the pending buffer's invalid gap [pending_len:64] from the stream: for
    # stream position j, source index is j while j < pending_len, else shifted to
    # skip past the gap.
    total_slots = (max_blocks + 1) * _BLOCK
    pos = jnp.arange(total_slots, dtype=jnp.int32)
    src_idx = pos + jnp.where(pos < pl, jnp.int32(0), _BLOCK - pl)
    src_idx = jnp.clip(src_idx, 0, combined_src.shape[0] - 1)
    combined = combined_src[src_idx]  # [total_slots], valid prefix [0:new_len]

    h = state.h.reshape(1, 8)
    for k in range(max_blocks):
        block = combined[k * _BLOCK : (k + 1) * _BLOCK]  # static slice [64]
        words = device_sha256.block_to_words(block.reshape(1, _BLOCK))
        h_new = device_sha256.compress(h, words)
        # Blocks past the live count are padding-only: leave the midstate untouched.
        h = jnp.where(jnp.int32(k) < active_blocks, h_new, h)

    tail_len = new_len - active_blocks * _BLOCK
    tail = jax.lax.dynamic_slice(combined, (active_blocks * _BLOCK,), (_BLOCK,))
    slot = jnp.arange(_BLOCK, dtype=jnp.int32)
    pending = jnp.where(slot < tail_len, tail, jnp.uint8(0))
    return Sha256State(
        h.reshape(8), pending, tail_len, state.total_len + jnp.int32(length)
    )


def sha256_stream_finalize(state: Sha256State, extras: Array) -> Array:
    """`SHA256(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
    static) — a non-mutating copy of the hash finished at the current length. One
    call finishes a whole batch of counter blocks (the transcript's counter-mode
    squeeze) sharing the base state. Returns uint8 [B, 32] big-endian digests.

    The trailing content is `pending[:pending_len] ‖ extras[b]` (≤ 63 + E bytes),
    so with the `0x80` byte and the 8-byte length it spans at most two blocks; the
    second block is compressed unconditionally and selected away when one suffices.
    """
    batch, e = extras.shape
    pl = state.pending_len
    content_len = pl + jnp.int32(e)
    msg_bytes = state.total_len + jnp.int32(e)
    # SHA-256's 64-bit length field; the high 32 bits are zero for any message
    # below 2**32 bits (512 MiB) — so the length is a uint32 and no x64 is needed.
    bitlen = msg_bytes.astype(jnp.uint32) * jnp.uint32(8)
    len_bytes = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)
    len_bytes = jnp.concatenate(
        [
            len_bytes,
            jnp.stack(
                [
                    ((bitlen >> jnp.uint32(24)) & jnp.uint32(0xFF)).astype(jnp.uint8),
                    ((bitlen >> jnp.uint32(16)) & jnp.uint32(0xFF)).astype(jnp.uint8),
                    ((bitlen >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(jnp.uint8),
                    (bitlen & jnp.uint32(0xFF)).astype(jnp.uint8),
                ]
            ),
        ]
    )  # [8] big-endian

    two_blocks = content_len > jnp.int32(55)  # need a 2nd block for pad + length?
    active_bytes = jnp.where(two_blocks, jnp.int32(128), jnp.int32(64))

    pos = jnp.arange(128, dtype=jnp.int32)
    # content = pending[:pl] ‖ extras[b], skipping the pending gap [pl:64].
    combined_src = jnp.concatenate(
        [jnp.broadcast_to(state.pending, (batch, _BLOCK)), extras.astype(jnp.uint8)],
        axis=1,
    )  # [B, 64+E]
    src_idx = jnp.clip(
        pos + jnp.where(pos < pl, jnp.int32(0), _BLOCK - pl), 0, _BLOCK + e - 1
    )
    content = combined_src[:, src_idx]  # [B, 128]

    is_content = (pos < content_len)[None, :]
    is_pad80 = (pos == content_len)[None, :]
    len_start = active_bytes - jnp.int32(8)
    is_len = ((pos >= len_start) & (pos < active_bytes))[None, :]
    len_val = len_bytes[jnp.clip(pos - len_start, 0, 7)][None, :]
    region = jnp.where(
        is_content,
        content,
        jnp.where(is_pad80, jnp.uint8(0x80), jnp.where(is_len, len_val, jnp.uint8(0))),
    )  # [B, 128]

    h = jnp.broadcast_to(state.h, (batch, 8))
    h1 = device_sha256.compress(h, device_sha256.block_to_words(region[:, 0:_BLOCK]))
    h2 = device_sha256.compress(h1, device_sha256.block_to_words(region[:, _BLOCK:128]))
    return device_sha256.serialize_digest(jnp.where(two_blocks, h2, h1))
