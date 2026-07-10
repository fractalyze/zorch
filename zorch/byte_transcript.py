# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-oriented Fiat-Shamir transcript: the `ByteTranscript` seam and a
Merlin-over-hash duplex (`ByteHashTranscript`) parameterized by a `ByteHash`.

This is the HOST-side, byte-oriented sibling of the device-resident algebraic
`DuplexTranscript` (`transcript.py`). The two form a taxonomy (see
`docs/transcript.md`): an algebraic sponge whose `observe`/`sample` are *device*
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
from functools import partial
from typing import TYPE_CHECKING, Protocol, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

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

# Proof-of-work: nonces a fused-search `lax.while_loop` step tests in parallel
# (static shape); a host hash uses window 1 (sequential early-exit, like
# hashlib). 2^12 balances the fused digest's latency floor (~0.15 ms/window on
# GPU) against hashes wasted past the hit: typical grinds are 2^10..2^14
# expected work, so low-bit grinds stay one cheap window and high-bit ones tile.
_GRIND_WINDOW = 1 << 12


def _len8(n: int) -> bytes:
    """A length / nonce as 8 little-endian bytes (the transcript's only integer
    encoding — fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


def _leading_zero_bits_ok(digests: np.ndarray | Array, bits: int) -> np.ndarray | Array:
    """Vectorized PoW predicate: whether each digest (uint8 `[B, digest_size]`) has
    >= `bits` leading zero bits, big-endian (digest[0] most significant).
    Array-agnostic (numpy or jax input) so the host `verify_pow` and the fused
    grind search share ONE definition and can never drift."""
    full, extra = divmod(bits, 8)
    ok = (digests[:, :full] == 0).all(axis=1)
    if extra:
        ok = ok & ((digests[:, full] >> np.uint8(8 - extra)) == 0)
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


class ByteGrindingTranscript(ByteTranscript, Protocol):
    """A `ByteTranscript` with a u64-nonce proof-of-work grind. Split from the
    base seam (as `transcript.GrindingTranscript` is) — the byte/nonce PoW is a
    different predicate from the field-element one and must not be cross-used."""

    def grind_pow(self, bits: int) -> tuple[Self, int]: ...
    def verify_pow(self, nonce: int, bits: int) -> tuple[Self, bool]: ...


@partial(jax.jit, static_argnums=(1, 2, 3))
def _fused_grind_search(
    state: np.ndarray, grind_hash: ByteHash, bits: int, window: int
) -> Array:
    """Device grind search, the byte-transcript twin of
    `DuplexTranscript._grind_search`: each `lax.while_loop` step hashes a
    `window`-wide nonce batch through the fused digest IN PARALLEL and keeps the
    lowest-index hit; the loop tiles windows (early-exiting on the first hit,
    which for typical `bits` is the first window) and caps `base` below the u64
    wrap so `base + window` stays in range. Candidate rows are
    `state || nonce_le8` — `bitcast_convert_type` iterates bytes LSB-first,
    matching `_len8`. The hit predicate is the shared
    `_leading_zero_bits_ok`."""
    # uint32 search, like `DuplexTranscript._grind_search`: runs with jax x64
    # off, and 2^32 nonces is far beyond any practical `bits`. `nonce_le8`'s
    # high four bytes are therefore constant zero.
    state_dev = jnp.asarray(state, dtype=jnp.uint8)
    offsets = jnp.arange(window, dtype=jnp.uint32)
    bound = jnp.uint32(2**32 - window)
    zeros_hi = jnp.zeros((window, 4), dtype=jnp.uint8)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        found, base, _ = carry
        return jnp.logical_and(jnp.logical_not(found), base < bound)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        found, base, best = carry
        nonces = base + offsets
        tail = lax.bitcast_convert_type(nonces, jnp.uint8)  # [window, 4] LSB-first
        rows = jnp.concatenate(
            [jnp.broadcast_to(state_dev, (window, state_dev.size)), tail, zeros_hi],
            axis=1,
        )
        hits = _leading_zero_bits_ok(grind_hash.digest(rows), bits)
        any_hit = jnp.any(hits)
        first = jnp.min(jnp.where(hits, offsets, jnp.uint32(window)))
        return (
            jnp.logical_or(found, any_hit),
            base + jnp.uint32(window),
            jnp.where(any_hit, base + first, best),
        )

    init = (jnp.bool_(False), jnp.uint32(0), jnp.uint32(0))
    _found, _base, nonce = lax.while_loop(cond, body, init)
    return nonce


@dataclass(frozen=True)
class ByteHashTranscript:
    """Merlin-style byte duplex over an injected `ByteHash`. Functional: every op
    returns a new transcript whose `buffer` is the running absorbed-byte stream. A
    host object (a `bytes` buffer, not a jit-traced pytree); the `ByteHash` chooses
    the squeeze substrate — `HostSha256` (host `hashlib`) or `Sha256` (the
    `zorch.sha256` device marker). Byte-identical whichever is injected."""

    buffer: bytes
    byte_hash: ByteHash
    # The PoW grind's hash, when it should differ from the stream's: the byte
    # stream is strictly sequential (a host hash wins — a per-absorb device hop
    # regresses it) while the grind is embarrassingly parallel (a fused hash
    # wins). None means "same as byte_hash". Both substrates hash to identical
    # bytes, so the split is invisible on the wire.
    grind_hash: ByteHash | None = None

    @property
    def has_dedicated_fusion(self) -> bool:
        # Delegates to the hash — as `DuplexSponge.has_dedicated_fusion` delegates
        # to its `Permutation`. Names no concrete hash.
        return self.byte_hash.has_dedicated_fusion

    @classmethod
    def new(
        cls,
        domain: bytes,
        byte_hash: ByteHash,
        grind_hash: ByteHash | None = None,
    ) -> ByteHashTranscript:
        """Seed with a length-prefixed domain so prefix domains can't collide:
        `[OP_DOMAIN] || len8(domain) || domain`."""
        return cls(
            bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain),
            byte_hash,
            grind_hash,
        )

    # ---- internal absorb / squeeze (over byte_hash.digest) ----
    def _absorb(self, payload: bytes) -> ByteHashTranscript:
        return ByteHashTranscript(
            self.buffer + payload, self.byte_hash, self.grind_hash
        )

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
        leading zero bits. A fused grind hash searches on device
        (`_fused_grind_search`, one dispatch); a host hash tests nonces
        sequentially with early exit. Both tile until a hit — unbounded, never
        returning an unchecked nonce."""
        grind_hash = self.grind_hash or self.byte_hash
        if grind_hash.has_dedicated_fusion:
            return int(
                _fused_grind_search(
                    np.frombuffer(state_digest, dtype=np.uint8),
                    grind_hash,
                    bits,
                    _GRIND_WINDOW,
                )
            )
        base = 0
        while True:
            row = np.frombuffer(state_digest + _len8(base), dtype=np.uint8)[None, :]
            if _leading_zero_bits_ok(np.asarray(grind_hash.digest(row)), bits)[0]:
                return base
            base += 1

    def grind_pow(self, bits: int) -> tuple[ByteHashTranscript, int]:
        """Lowest u64 nonce whose PoW passes (0 if bits==0), then absorb it via
        `observe_bytes` so subsequent challenges bind to it."""
        _validate_pow_bits(bits, self.byte_hash.digest_size)
        nonce = 0 if bits == 0 else self._grind(self._digest(), bits)
        return self.observe_bytes(_len8(nonce)), nonce

    def verify_pow(self, nonce: int, bits: int) -> tuple[ByteHashTranscript, bool]:
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
    # Seam-conformance pins (docs/conventions.md).
    _bt: type[ByteTranscript] = ByteHashTranscript
    _bg: type[ByteGrindingTranscript] = ByteHashTranscript
