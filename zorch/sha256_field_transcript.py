# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-element `Transcript` surface over the streaming SHA-256 core.

The SHA-256 sibling of the algebraic `transcript.DuplexTranscript`, and shaped
like it: a frozen pytree dataclass whose methods are plain traced state
transitions — jit-first, scan-threadable, no host substrate anywhere. It keeps
the Merlin byte framing (op tag, u64-LE count, `SHA256(buffer ‖ ctr)`
counter-squeeze, re-absorb) on the fixed-shape `Sha256State`
(`hash/sha256.py`), so a slice observe / sample is byte-identical to
`ByteHashTranscript`'s `observe_slice` / `sample_slice`, and the proof-of-work
grind is `grind`/`check_witness` with `DuplexTranscript`'s exact semantics
(device windowed search via `zorch.grind`; the advanced transcript absorbs the
witness regardless, and `check_witness` is the soundness gate).

Scheme-agnostic: `dtype` is the challenge element's (scalar) type. `observe`
bitcasts values to bytes and `sample` reinterprets squeezed bytes back, so the
element width is `dtype.itemsize`. A binary-field element wider than a scalar
dtype (e.g. a `uint64[2]` pair) rides the sumcheck only through a field-ops
seam the consumer supplies; a byte-framed challenger can use
`ByteHashTranscript` instead.

The observe/sample surface is exactly the Merlin wire's op vocabulary — one
method per op tag, because the tag is transcript-semantic (two ops with the
same payload and different tags produce different challenge streams), and a
mode flag would be the same arity hidden in an argument:

    observe(values)        [OP_OBSERVE, KIND_SLICE]  count-prefixed vector
    observe_scalar(value)  [OP_OBSERVE, KIND_SCALAR] per element, no prefix
    observe_label(label)   [OP_LABEL]                domain separation
    observe_bytes(data)    [OP_BYTES]                opaque bytes (roots, PoW)
    sample(n)              [OP_SQUEEZE, KIND_SLICE]  count-prefixed squeeze
    sample_scalar()        [OP_SQUEEZE, KIND_SCALAR] one-element squeeze
"""
from __future__ import annotations

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
from zorch.grind import GRIND_WINDOW, grind_search
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


def _u32_le_bytes(values: Array) -> Array:
    """uint32 `[...]` -> uint8 `[..., 4]`, little-endian."""
    return lax.bitcast_convert_type(values, jnp.uint8)


def _leading_zero_bits_ok(digests: Array, bits: int) -> Array:
    """Whether each digest (uint8 `[B, 32]`) has >= `bits` leading zero bits,
    big-endian (digest[..., 0] most significant). Traceable; byte-identical to
    `byte_transcript._leading_zero_bits_ok`."""
    full, extra = divmod(bits, 8)
    ok = jnp.all(digests[:, :full] == 0, axis=1)
    if extra:
        ok = ok & ((digests[:, full] >> np.uint8(8 - extra)) == 0)
    return ok


@partial(register_dataclass, data_fields=["state"], meta_fields=["dtype"])
@dataclass(frozen=True)
class Sha256FieldTranscript:
    """Device SHA-256 transcript satisfying `transcript.Transcript`, threadable
    through a `lax.scan` / `@jit` like `DuplexTranscript`. State is the
    streaming `Sha256State` pytree; `dtype` (static) is the challenge element
    type."""

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
        `SHA256(buffer ‖ ctr)` for ctr=0,1,…, one batched finalize."""
        nblocks = (nbytes + 31) // 32
        extras = _const_u8(b"".join(_len8(ctr) for ctr in range(nblocks))).reshape(
            nblocks, 8
        )
        digs = sha256_stream_finalize(self.state, extras)  # [nblocks, 32]
        return digs.reshape(-1)[:nbytes]

    def observe(self, values: Array) -> Sha256FieldTranscript:
        """Absorb `values` under slice framing: `[OP_OBSERVE, KIND_SLICE] ||
        len8(count) || serialized bytes`. Byte-identical to the byte
        transcript's `observe_slice` of the same serialized bytes."""
        vals_u8 = self._elem_bytes(values).reshape(-1)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(jnp.concatenate([framing, vals_u8]))

    def observe_scalar(self, value: Array) -> Sha256FieldTranscript:
        """Absorb under scalar framing `[OP_OBSERVE, KIND_SCALAR] || elem_bytes`
        — no length prefix, a scalar's width being implicit in the dtype. A 0-d
        `value` is one op; an `[n]` array is n ops (one per element, in order),
        built as ONE absorb payload. Byte-identical to the byte transcript's
        `observe_scalar` per element; distinct from `observe` (the KIND tag
        differs)."""
        vals_u8 = self._elem_bytes(value).reshape(-1, self._item_bytes())
        framing = jnp.broadcast_to(
            _const_u8(bytes([OP_OBSERVE, KIND_SCALAR])), (vals_u8.shape[0], 2)
        )
        return self._absorb(jnp.concatenate([framing, vals_u8], axis=1).reshape(-1))

    def observe_label(self, label: bytes) -> Sha256FieldTranscript:
        """Absorb a domain-separation label `[OP_LABEL] || len8(len) || label`.
        A compile-time host constant (labels are literals), so the whole absorb
        is one constant payload. Byte-identical to the byte transcript."""
        return self._absorb(
            _const_u8(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))
        )

    def observe_bytes(self, data: Array) -> Sha256FieldTranscript:
        """Absorb opaque bytes (e.g. a Merkle root computed on-device) under
        `[OP_BYTES] || len8(len) || data`. `data` is a uint8 array whose length
        is static (it rides the framing prefix). Byte-identical to the byte
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
        KIND_SCALAR]`, counter-squeeze `itemsize` bytes, re-absorb, reinterpret
        to one `dtype` element (0-D). Byte-identical to the byte transcript's
        `sample_scalar`; distinct from `sample(1)` (the KIND tag differs)."""
        t = self._absorb(_const_u8(bytes([OP_SQUEEZE, KIND_SCALAR])))
        squeezed = t._squeeze_bytes(self._item_bytes())
        t = t._absorb(squeezed)
        return t, self._u8_to_elems(squeezed, 1)[0]

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Sha256FieldTranscript, Array]:
        return self.observe(values).sample(n)

    # ---- proof-of-work (DuplexTranscript's grind/check_witness shape) ----
    def _pow_state(self) -> Sha256State:
        """A fresh stream over the PoW state digest `SHA256(buffer)`: candidate
        digests are `finalize(pow_state, counter_le8)` batches. Matches the byte
        transcript's `HASH(state_digest || nonce_le8)`."""
        digest = sha256_stream_finalize(self.state, jnp.zeros((1, 0), dtype=jnp.uint8))[
            0
        ]
        return sha256_stream_absorb(sha256_stream_init(), digest)

    def _witness_bytes(self, witness: Array) -> Array:
        """The u64-LE nonce wire bytes of a uint32 witness (high 4 bytes zero —
        the search domain is uint32, like `DuplexTranscript._grind_search`)."""
        lo4 = _u32_le_bytes(jnp.asarray(witness, jnp.uint32).reshape(1))[0]
        return jnp.concatenate([lo4, jnp.zeros(4, jnp.uint8)])

    def _absorb_witness(self, witness: Array) -> Sha256FieldTranscript:
        framing = _const_u8(bytes([OP_BYTES]) + _len8(8))
        return self._absorb(jnp.concatenate([framing, self._witness_bytes(witness)]))

    def grind(
        self, pow_bits: int, *, chunk: int = GRIND_WINDOW
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Find a proof-of-work witness — the lowest nonce whose
        `SHA256(state_digest || nonce_le8)` has `pow_bits` leading zero bits —
        and return the transcript advanced past it (the nonce absorbed under
        flock's `OP_BYTES` wire), plus the witness. Fully traceable
        (`zorch.grind.grind_search` windowed device search); does not raise on
        an exhausted search: `check_witness` is the soundness gate, so which
        witness the search returns is soundness-neutral."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        if pow_bits == 0:
            # No work required: the canonical zero witness always passes.
            witness = jnp.zeros((), jnp.uint32)
            return self._absorb_witness(witness), witness
        pow_state = self._pow_state()

        def check_batch(counters: Array) -> Array:
            nonce8 = jnp.concatenate(
                [_u32_le_bytes(counters), jnp.zeros((counters.shape[0], 4), jnp.uint8)],
                axis=1,
            )
            return _leading_zero_bits_ok(
                sha256_stream_finalize(pow_state, nonce8), pow_bits
            )

        witness = grind_search(check_batch, 2**32, chunk)
        return self._absorb_witness(witness), witness

    def check_witness(
        self, witness: Array, pow_bits: int
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Verifier mirror of `grind`: check the PoW (`pow_bits == 0` requires
        the canonical witness 0), then absorb the witness REGARDLESS so the
        transcript stays in lockstep. Returns the advanced transcript and the
        device boolean verdict."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        witness = jnp.asarray(witness, jnp.uint32).reshape(())
        if pow_bits == 0:
            ok = witness == jnp.uint32(0)
        else:
            nonce8 = self._witness_bytes(witness)[None, :]
            digs = sha256_stream_finalize(self._pow_state(), nonce8)
            ok = _leading_zero_bits_ok(digs, pow_bits)[0]
        return self._absorb_witness(witness), ok

    # ---- element <-> byte serde ----
    def _elem_bytes(self, values: Array) -> Array:
        """Element array -> uint8 `[..., itemsize]` — a direct bitcast to bytes.
        (The wide-binary-field <-> uint8 bitcast once miscompiled on the CPU
        PJRT backend, flock-zorch#75, forcing a uint32-lane detour; that is
        fixed as of the dev20260713 stack.)"""
        return lax.bitcast_convert_type(values, jnp.uint8)

    def _u8_to_elems(self, u8: Array, n: int) -> Array:
        """Flat uint8 `[n * itemsize]` -> `[n]` `dtype` elements (inverse of
        `_elem_bytes`)."""
        return lax.bitcast_convert_type(
            u8.reshape(n, self._item_bytes()), self.dtype
        ).reshape(n)
