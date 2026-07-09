# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-element `Transcript` surface over the streaming SHA-256 core.

`zorch.sumcheck.prove` threads a transcript through a `lax.scan` via
`observe(Array)` / `sample(n)` / `observe_and_sample`. `Sha256FieldTranscript`
implements that protocol on the fixed-shape `Sha256State` (`hash/sha256.py`), in
pure JAX, so it is a valid scan carry — the device SHA-256 sibling of the algebraic
`transcript.DuplexTranscript`. It keeps the same Merlin slice framing as the byte
transcript (op tag, u64-LE count, `SHA256(buffer ‖ ctr)` counter-squeeze,
re-absorb), so a slice observe / sample is byte-identical to `ByteHashTranscript`'s
`observe_slice` / `sample_slice` — but now scan-threadable, collapsing a byte
Fiat-Shamir round loop into one device program.

Scheme-agnostic: `dtype` is the challenge element's (scalar) type. `observe`
bitcasts values to bytes and `sample` reinterprets squeezed bytes back, so the
element width is `dtype.itemsize`. A binary-field element wider than a scalar dtype
(e.g. a `uint64[2]` pair) rides the sumcheck only through a field-ops seam the
consumer supplies; a byte-framed challenger can use `ByteHashTranscript` instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

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
    _validate_pow_bits,
)
from zorch.hash.sha256 import (
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)

# SHA-256 digest width — the PoW state digest and every squeeze block are 32 B.
_DIGEST_BYTES = 32


def _len8(n: int) -> bytes:
    """A length / count as 8 little-endian bytes (the transcript's only integer
    encoding — fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


def _const_u8(data: bytes) -> Array:
    """A compile-time-constant byte payload as a device uint8 array."""
    return jnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _leading_zero_bits_ok(digest: bytes, bits: int) -> bool:
    """Whether `digest` has >= `bits` leading zero bits, big-endian. The host PoW
    predicate — byte-identical to `byte_transcript._leading_zero_bits_ok`; the
    grind oracle test pins the two together."""
    full, extra = divmod(bits, 8)
    if any(digest[:full]):
        return False
    return extra == 0 or (digest[full] >> (8 - extra)) == 0


def _grind_host(state_digest: bytes, bits: int) -> int:
    """Lowest u64 nonce with `SHA256(state_digest || nonce_le8)` having `bits`
    leading zero bits — a host sequential search (hashlib), byte-identical to
    `ByteHashTranscript`'s host grind (window 1). Unbounded; never returns an
    unchecked nonce."""
    nonce = 0
    while True:
        d = hashlib.sha256(state_digest + _len8(nonce)).digest()
        if _leading_zero_bits_ok(d, bits):
            return nonce
        nonce += 1


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

    def _elems_to_u8(self, values: Array) -> Array:
        """Element array -> flat uint8, routed through uint32 lanes. The direct
        wide-binary-field <-> uint8 bitcast miscompiles on the CPU PJRT backend
        (flock-zorch#75); uint32 is the dtype's native lane width and is correct
        on both backends. A uint32-native challenge dtype takes the first bitcast
        as the identity, so the routing is a no-op there."""
        u32 = lax.bitcast_convert_type(values, jnp.uint32)
        return lax.bitcast_convert_type(u32, jnp.uint8).reshape(-1)

    def _u8_to_elems(self, u8: Array, n: int) -> Array:
        """Flat uint8 `[n * itemsize]` -> `[n]` `dtype` elements, via uint32 lanes
        (the inverse of `_elems_to_u8`; CPU-safe, flock-zorch#75). `bitcast` groups
        exactly the trailing 4 bytes into one uint32, so the byte axis is `4` (a
        `[n, lanes, 4]` view), not the full `itemsize`."""
        lanes = self._item_bytes() // 4
        u32 = lax.bitcast_convert_type(
            u8.reshape(n, lanes, 4), jnp.uint32
        )  # [n, lanes]
        return lax.bitcast_convert_type(u32, self.dtype).reshape(n)

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
        vals_u8 = self._elems_to_u8(values)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(jnp.concatenate([framing, vals_u8]))

    def observe_scalar(self, value: Array) -> Sha256FieldTranscript:
        """Absorb one element under scalar framing `[OP_OBSERVE, KIND_SCALAR] ||
        elem_bytes` — no length prefix, a scalar's width being implicit in the
        dtype. Byte-identical to the byte transcript's `observe_scalar`; distinct
        from `observe` of a length-1 slice (the KIND tag differs)."""
        vbytes = self._elems_to_u8(value)
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SCALAR]))
        return self._absorb(jnp.concatenate([framing, vbytes]))

    def observe_label(self, label: bytes) -> Sha256FieldTranscript:
        """Absorb a domain-separation label `[OP_LABEL] || len8(len) || label`.
        A compile-time host constant (labels are literals), so the whole absorb
        is one constant payload. Byte-identical to the byte transcript."""
        return self._absorb(
            _const_u8(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))
        )

    def observe_bytes(self, data: Array) -> Sha256FieldTranscript:
        """Absorb opaque bytes (e.g. a Merkle root computed on-device) under
        `[OP_BYTES] || len8(len) || data`. `data` is a uint8 array whose length is
        static (it rides the framing prefix). Byte-identical to the byte
        transcript's `observe_bytes` of the same bytes."""
        data = jnp.asarray(data, jnp.uint8).reshape(-1)
        framing = _const_u8(bytes([OP_BYTES]) + _len8(int(data.shape[0])))
        return self._absorb(jnp.concatenate([framing, data]))

    def sample(self, n: int = 1) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze `n` challenge elements: absorb `[OP_SQUEEZE, KIND_SLICE] ||
        len8(n)`, counter-squeeze `n * itemsize` bytes, re-absorb them, and
        reinterpret to `n` elements of `dtype`."""
        t = self._absorb(_const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(n)))
        squeezed = t._squeeze_bytes(n * self._item_bytes())
        t = t._absorb(squeezed)
        return t, self._u8_to_elems(squeezed, n)

    def sample_scalar(self) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze one challenge under scalar framing: absorb `[OP_SQUEEZE,
        KIND_SCALAR]`, counter-squeeze `itemsize` bytes, re-absorb, reinterpret to
        one `dtype` element (0-D). Byte-identical to the byte transcript's
        `sample_scalar`; distinct from `sample(1)` (the KIND tag differs)."""
        t = self._absorb(_const_u8(bytes([OP_SQUEEZE, KIND_SCALAR])))
        squeezed = t._squeeze_bytes(self._item_bytes())
        t = t._absorb(squeezed)
        return t, self._u8_to_elems(squeezed, 1)[0]

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Sha256FieldTranscript, Array]:
        return self.observe(values).sample(n)

    # ---- proof-of-work (host to start; a device grind is a follow-up) ----
    def _state_digest(self) -> bytes:
        """`SHA256(buffer)` — the PoW base: a non-mutating finalize of the current
        streaming state (no counter, no extras), materialized to host bytes for the
        host grind. Matches the byte transcript's `_digest`."""
        dig = sha256_stream_finalize(self.state, jnp.zeros((1, 0), dtype=jnp.uint8))
        return bytes(np.asarray(dig)[0])

    def grind_pow(self, bits: int) -> tuple[Sha256FieldTranscript, int]:
        """Host proof-of-work grind: the lowest u64 nonce whose PoW passes (0 when
        `bits == 0`), absorbed via `observe_bytes` so later challenges bind to it.
        Host to start — a one-shot search, not per-round threading, so it does not
        break device graph capture; a device grind is a follow-up. Byte-identical
        to `ByteHashTranscript.grind_pow`."""
        _validate_pow_bits(bits, _DIGEST_BYTES)
        nonce = 0 if bits == 0 else _grind_host(self._state_digest(), bits)
        return self.observe_bytes(_const_u8(_len8(nonce))), nonce

    def verify_pow(self, nonce: int, bits: int) -> tuple[Sha256FieldTranscript, bool]:
        """Verifier mirror of `grind_pow`: check the PoW (`bits == 0` requires the
        canonical nonce 0), then absorb the nonce REGARDLESS so the transcript stays
        in lockstep. Byte-identical to `ByteHashTranscript.verify_pow`."""
        _validate_pow_bits(bits, _DIGEST_BYTES)
        if bits == 0:
            ok = nonce == 0
        else:
            d = hashlib.sha256(self._state_digest() + _len8(nonce)).digest()
            ok = _leading_zero_bits_ok(d, bits)
        return self.observe_bytes(_const_u8(_len8(nonce))), ok
