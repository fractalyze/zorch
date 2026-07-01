# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device sibling of `byte_transcript.Sha256Transcript`.

Same Merlin-over-SHA-256 byte framing (op tags, u64-LE length prefixes,
`SHA256(buffer ‖ ctr)` counter-squeeze, re-absorb of the squeezed bytes), but the
SHA-256 compression runs ON DEVICE via the name-routed `zorch.sha256` marker
(`zorch.hash.sha256.digest`) instead of host `hashlib` — a byte Fiat-Shamir chain
that lowers to a GPU kernel, unlike the host `Sha256Transcript`. It therefore
reports `has_dedicated_fusion = True` (the device-byte row of the
`docs/transcript.md` taxonomy).

Byte-identical to the host transcript by construction — the absorbed-byte stream
uses the identical framing, and `zorch.hash.sha256.digest` is byte-identical to
`hashlib.sha256` (pinned by `hash/testing/sha256_test.py`).

This class keeps the host's growing-`bytes` buffer and re-hashes it per squeeze:
device-hashed, but not a fixed-shape `lax.scan` carry. The streaming `Sha256State`
below (`h[8]` + `<64 B` pending + length) is the fixed-shape core
`Sha256FieldTranscript` uses to thread `@jit` / `lax.scan`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
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


# Proof-of-work grind: the candidate window each device batch tests at once. A
# `bits`-hit lands in the first window for any practical `pow_bits`, so the search
# is normally one marker call; larger `pow_bits` just tile more windows (never a
# silent cap — the search keeps going, matching the host transcript).
_GRIND_WINDOW = 1 << 16


def _leading_zero_bits_ok(digests: np.ndarray, bits: int) -> np.ndarray:
    """Whether each digest in `digests` (uint8 [B, 32]) has ≥ `bits` leading zero
    bits, big-endian (digest[0] most significant) — the proof-of-work predicate,
    vectorized over the batch."""
    full, extra = divmod(bits, 8)
    ok = np.all(digests[:, :full] == 0, axis=1)
    if extra:
        ok &= (digests[:, full] >> np.uint8(8 - extra)) == 0
    return ok


def _grind_search(state_digest: bytes, bits: int, window: int) -> int:
    """Lowest u64 nonce with `SHA256(state_digest ‖ nonce_le8)` having `bits`
    leading zero bits. Tests a whole `window` of nonces per device digest call and
    returns the first hit; tiles windows until one lands (unbounded, like the host
    — never returns an unchecked nonce)."""
    prefix = np.frombuffer(state_digest, dtype=np.uint8)  # [32]
    base = 0
    while True:
        nonces = np.arange(base, base + window, dtype="<u8")
        nonce_bytes = nonces.view(np.uint8).reshape(
            window, 8
        )  # le8, little-endian host
        msgs = np.concatenate(
            [np.broadcast_to(prefix, (window, prefix.shape[0])), nonce_bytes], axis=1
        )  # [window, 40]
        digs = np.asarray(device_sha256.digest(msgs))  # [window, 32]
        hits = np.flatnonzero(_leading_zero_bits_ok(digs, bits))
        if hits.size:
            return int(base + hits[0])
        base += window


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

    # ---- proof-of-work (byte/nonce PoW, identical predicate to the host) ----
    def grind_pow(
        self, bits: int, *, window: int = _GRIND_WINDOW
    ) -> tuple[DeviceSha256Transcript, int]:
        """Lowest u64 nonce with `SHA256(state_digest ‖ nonce_le8)` having `bits`
        leading zero bits (0 if bits==0), then absorb the nonce via `observe_bytes`
        so subsequent challenges bind to it. The search is device-batched (one
        marker call per `window` candidates) but returns the same lowest nonce the
        host finds — byte-identical."""
        if bits == 0:
            return self.observe_bytes(_len8(0)), 0
        nonce = _grind_search(self._digest(), bits, window)
        return self.observe_bytes(_len8(nonce)), nonce

    def verify_pow(self, nonce: int, bits: int) -> tuple[DeviceSha256Transcript, bool]:
        """Verifier mirror: check the PoW (bits==0 requires the canonical nonce 0),
        then absorb the nonce REGARDLESS so the transcript stays in lockstep."""
        if bits == 0:
            ok = nonce == 0
        else:
            dig = _device_digest(self._digest() + _len8(nonce))
            ok = bool(
                _leading_zero_bits_ok(
                    np.frombuffer(dig, dtype=np.uint8)[None, :], bits
                )[0]
            )
        return self.observe_bytes(_len8(nonce)), ok


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
# bytes exactly, incrementally. `Sha256FieldTranscript` builds the field-element
# `Transcript` surface on this, so a byte Fiat-Shamir round loop folds through one
# device program.
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


# ---------------------------------------------------------------------------
# Field-element surface: zorch's `transcript.Transcript` over the streaming core.
#
# `zorch.sumcheck.prove` threads a transcript through a `lax.scan` via
# `observe(Array)` / `sample(n)` / `observe_and_sample`. `Sha256FieldTranscript`
# implements that protocol on the fixed-shape `Sha256State`, in pure JAX, so it is
# a valid scan carry — the device SHA-256 sibling of `transcript.DuplexTranscript`.
# It keeps the same Merlin slice framing as the byte transcript (op tag, u64-LE
# count, `SHA256(buffer ‖ ctr)` counter-squeeze, re-absorb), so a slice observe /
# sample is byte-identical to `DeviceSha256Transcript.observe_slice` /
# `sample_slice` — but now scan-threadable.
#
# Scheme-agnostic: `dtype` is the challenge element's (scalar) type. `observe`
# bitcasts values to bytes and `sample` reinterprets squeezed bytes back, so the
# element width is `dtype.itemsize`. A binary-field element wider than a scalar
# dtype (e.g. a `uint64[2]` pair) rides the sumcheck only through a field-ops seam
# the consumer supplies; a byte-framed challenger can use the byte surface above.
# ---------------------------------------------------------------------------


def _const_u8(data: bytes) -> Array:
    """A compile-time-constant byte payload as a device uint8 array."""
    return jnp.asarray(np.frombuffer(data, dtype=np.uint8))


@partial(register_dataclass, data_fields=["state"], meta_fields=["dtype"])
@dataclass(frozen=True)
class Sha256FieldTranscript:
    """Device SHA-256 transcript satisfying `transcript.Transcript`, threadable
    through `zorch.sumcheck.prove` under `@jit`. State is the streaming
    `Sha256State` pytree; `dtype` (static) is the challenge element type."""

    state: Sha256State
    dtype: Any

    @property
    def has_dedicated_fusion(self) -> bool:
        # The SHA-256 chain lowers to a GPU kernel via the zorch.sha256 marker.
        return True

    @classmethod
    def new(cls, domain: bytes, dtype: Any) -> Sha256FieldTranscript:
        seed = _const_u8(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))
        return cls(sha256_stream_absorb(sha256_stream_init(), seed), np.dtype(dtype))

    def _item_bytes(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    def _absorb(self, payload: Array) -> Sha256FieldTranscript:
        return replace(self, state=sha256_stream_absorb(self.state, payload))

    def _squeeze_bytes(self, nbytes: int) -> Array:
        """`nbytes` counter-squeeze bytes from the CURRENT state (no mutation) —
        `SHA256(buffer ‖ ctr_le8)` for ctr=0,1,…, one batched finalize."""
        nblocks = (nbytes + 31) // 32
        extras = _const_u8(b"".join(_len8(ctr) for ctr in range(nblocks))).reshape(
            nblocks, 8
        )
        digs = sha256_stream_finalize(self.state, extras)  # [nblocks, 32]
        return digs.reshape(-1)[:nbytes]

    def observe(self, values: Array) -> Sha256FieldTranscript:
        """Absorb `values` under slice framing: `[OP_OBSERVE, KIND_SLICE] ||
        len8(count) || lo‖hi-serialized bytes`. Byte-identical to the byte
        transcript's `observe_slice` of the same serialized bytes."""
        vals_u8 = lax.bitcast_convert_type(values, jnp.uint8).reshape(-1)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(jnp.concatenate([framing, vals_u8]))

    def sample(self, n: int = 1) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze `n` challenge elements: absorb `[OP_SQUEEZE, KIND_SLICE] ||
        len8(n)`, counter-squeeze `n * itemsize` bytes, re-absorb them, and
        reinterpret to `n` elements of `dtype`."""
        t = self._absorb(_const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(n)))
        squeezed = t._squeeze_bytes(n * self._item_bytes())
        t = t._absorb(squeezed)
        return t, squeezed.view(self.dtype)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Sha256FieldTranscript, Array]:
        return self.observe(values).sample(n)
