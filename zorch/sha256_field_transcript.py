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
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
import numpy as np
from frx import Array, jit, lax
from frx.tree_util import register_dataclass
from hash_frx.fusion import fused_region
from hash_frx.sha256 import (
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)

from zorch.byte_transcript import (
    KIND_SCALAR,
    KIND_SLICE,
    OP_BYTES,
    OP_DOMAIN,
    OP_LABEL,
    OP_OBSERVE,
    OP_SQUEEZE,
    _len8,
    _validate_pow_bits,
)
from zorch.grind import grind_search, grind_window_for, leading_zero_bits_ok

# SHA-256 digest width — the PoW state digest and every squeeze block are 32 B.
_DIGEST_BYTES = 32


def _const_u8(data: bytes) -> Array:
    """A compile-time-constant byte payload as a device uint8 array."""
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _u32_le_bytes(values: Array) -> Array:
    """uint32 `[...]` -> uint8 `[..., 4]`, little-endian."""
    return lax.bitcast_convert_type(values, fnp.uint8)


# ============================================================================
# Squeeze-hop fusion marker. `hash_frx.sha256` marks the COMPRESSION; this marks
# the whole squeeze around it — the same layering `zorch.duplex_fs` adds over
# `hash_frx.poseidon2`, for the same reason.
#
# A squeeze is `absorb(framing) -> counter-squeeze -> re-absorb`. The streaming
# state is branchless, so each absorb compresses speculatively and selects and
# finalize emits both padding candidates: four `hash_frx.sha256` regions, each a
# fusion barrier, so the scalar bookkeeping between them cannot merge either.
# That is ~14 GPU launches per `sample_scalar`, and on a latency-bound prove the
# cost is the launch count, not the arithmetic.
#
# The decomposition is the plain `_squeeze_hop`, so with no emitter the marker
# inlines byte-identically. `pending_len` / `total_len` ride as runtime OPERANDS
# (the packed `counts` leaf), not attrs, so the data-dependent tests are uniform
# across the single thread and one recompile-free kernel serves every position.
# ============================================================================
SHA256_SQUEEZE_MARKER = "zorch.sha256_squeeze"
SHA256_SQUEEZE_MARKER_VERSION = 1


def _squeeze_hop(
    state: Sha256State, framing: Array, nbytes: int
) -> tuple[Sha256State, Array]:
    """The plain squeeze hop: absorb `framing`, counter-squeeze `nbytes` bytes
    from the resulting state (`SHA256(buffer ‖ ctr)` for ctr=0,1,… as one batched
    finalize), then re-absorb them. Returns the advanced state and the raw
    squeezed bytes; reinterpreting those to field elements stays with the caller,
    so the region carries no dtype."""
    absorbed = sha256_stream_absorb(state, framing)
    nblocks = (nbytes + _DIGEST_BYTES - 1) // _DIGEST_BYTES
    extras = _const_u8(b"".join(_len8(ctr) for ctr in range(nblocks))).reshape(
        nblocks, 8
    )
    squeezed = sha256_stream_finalize(absorbed, extras).reshape(-1)[:nbytes]
    return sha256_stream_absorb(absorbed, squeezed), squeezed


def _sha256_squeeze_region(
    h: Array,
    pending: Array,
    counts: Array,
    framing: Array,
    *,
    nbytes: int,
) -> tuple[Array, Array, Array, Array]:
    """The `zorch.sha256_squeeze` decomposition, entered at the runtime stream
    position carried in the state leaves. `nbytes` rides as an attr (the
    emitter's static squeeze width); the state stays in operands so one kernel
    serves any `pending_len`."""
    state, squeezed = _squeeze_hop(Sha256State(h, pending, counts), framing, nbytes)
    return (state.h, state.pending, state.counts, squeezed)


@partial(jit, static_argnames=("nbytes",), inline=True)
def _sha256_squeeze_zone(
    state: Sha256State, framing: Array, nbytes: int
) -> tuple[Sha256State, Array]:
    """The marked hop as one compiled dispatch carrying the
    `zorch.sha256_squeeze` composite, so an eager caller fires a single fused FS
    kernel. `inline=True` keeps a call site already inside an outer jit
    byte-identical (mirrors `transcript._duplex_fs_zone`)."""
    if nbytes == 0:
        return _squeeze_hop(state, framing, nbytes)
    h, pending, counts, squeezed = fused_region(
        _sha256_squeeze_region,
        state.h,
        state.pending,
        state.counts,
        framing,
        name=SHA256_SQUEEZE_MARKER,
        version=SHA256_SQUEEZE_MARKER_VERSION,
        nbytes=nbytes,
    )
    return Sha256State(h, pending, counts), squeezed


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
    def field(self) -> Any:
        return self.dtype

    @property
    def has_dedicated_fusion(self) -> bool:
        # The COMPRESSION lowers to a GPU kernel via the hash_frx.sha256 marker.
        # Says nothing about the hop above it: a squeeze also carries the
        # zorch.sha256_squeeze marker, which only fuses where a vendor emits it.
        return True

    @classmethod
    def new(cls, domain: bytes, dtype: Any) -> Sha256FieldTranscript:
        seed = _const_u8(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))
        return cls(sha256_stream_absorb(sha256_stream_init(), seed), np.dtype(dtype))

    def _item_bytes(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    def _absorb(self, payload: Array) -> Sha256FieldTranscript:
        return replace(self, state=sha256_stream_absorb(self.state, payload))

    def observe(self, values: Array) -> Sha256FieldTranscript:
        """Absorb `values` under slice framing: `[OP_OBSERVE, KIND_SLICE] ||
        len8(count) || serialized bytes`. Byte-identical to the byte
        transcript's `observe_slice` of the same serialized bytes."""
        vals_u8 = self._elem_bytes(values).reshape(-1)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(fnp.concatenate([framing, vals_u8]))

    def observe_scalar(self, value: Array) -> Sha256FieldTranscript:
        """Absorb under scalar framing `[OP_OBSERVE, KIND_SCALAR] || elem_bytes`
        — no length prefix, a scalar's width being implicit in the dtype. A 0-d
        `value` is one op; an `[n]` array is n ops (one per element, in order),
        built as ONE absorb payload. Byte-identical to the byte transcript's
        `observe_scalar` per element; distinct from `observe` (the KIND tag
        differs)."""
        return self._absorb(self._scalar_observe_wire(value))

    def _scalar_observe_wire(self, value: Array) -> Array:
        """`observe_scalar`'s payload: `[OP_OBSERVE, KIND_SCALAR] || elem_bytes`
        per element, in order. Split out so `observe_scalar_and_sample` can put
        the same bytes on the stream as part of a draw's framing."""
        vals_u8 = self._elem_bytes(value).reshape(-1, self._item_bytes())
        framing = fnp.broadcast_to(
            _const_u8(bytes([OP_OBSERVE, KIND_SCALAR])), (vals_u8.shape[0], 2)
        )
        return fnp.concatenate([framing, vals_u8], axis=1).reshape(-1)

    def _sample_scalar_after(
        self, payload: Array
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Put `payload` on the stream and draw a scalar, as ONE marked region —
        the BLAKE3 row's `_sample_scalar_after`, on this wire."""
        framing = fnp.concatenate(
            [payload, _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))]
        )
        state, squeezed = _sha256_squeeze_zone(self.state, framing, self._item_bytes())
        return replace(self, state=state), self._u8_to_elems(squeezed, 1)[0]

    def observe_scalar_and_sample(
        self, value: Array
    ) -> tuple[Sha256FieldTranscript, Array]:
        """`observe_scalar` then `sample_scalar`, as one marked region."""
        return self._sample_scalar_after(self._scalar_observe_wire(value))

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
        data = fnp.asarray(data, fnp.uint8).reshape(-1)
        framing = _const_u8(bytes([OP_BYTES]) + _len8(int(data.shape[0])))
        return self._absorb(fnp.concatenate([framing, data]))

    def sample(self, n: int = 1) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze `n` challenge elements: absorb `[OP_SQUEEZE, KIND_SLICE] ||
        len8(n)`, counter-squeeze `n * itemsize` bytes, re-absorb them, and
        reinterpret to `n` elements of `dtype`."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(n))
        state, squeezed = _sha256_squeeze_zone(
            self.state, framing, n * self._item_bytes()
        )
        return replace(self, state=state), self._u8_to_elems(squeezed, n)

    def sample_scalar(self) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze one challenge under scalar framing: absorb `[OP_SQUEEZE,
        KIND_SCALAR]`, counter-squeeze `itemsize` bytes, re-absorb, reinterpret
        to one `dtype` element (0-D). Byte-identical to the byte transcript's
        `sample_scalar`; distinct from `sample(1)` (the KIND tag differs)."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))
        state, squeezed = _sha256_squeeze_zone(self.state, framing, self._item_bytes())
        return replace(self, state=state), self._u8_to_elems(squeezed, 1)[0]

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Sha256FieldTranscript, Array]:
        return self.observe(values).sample(n)

    # ---- proof-of-work (DuplexTranscript's grind/check_witness shape) ----
    def _pow_state(self) -> Sha256State:
        """A fresh stream over the PoW state digest `SHA256(buffer)`: candidate
        digests are `finalize(pow_state, counter_le8)` batches. Matches the byte
        transcript's `HASH(state_digest || nonce_le8)`."""
        digest = sha256_stream_finalize(self.state, fnp.zeros((1, 0), dtype=fnp.uint8))[
            0
        ]
        return sha256_stream_absorb(sha256_stream_init(), digest)

    def _witness_bytes(self, witness: Array) -> Array:
        """The u64-LE nonce wire bytes of a uint32 witness (high 4 bytes zero —
        the search domain is uint32, like `DuplexTranscript._grind_search`)."""
        lo4 = _u32_le_bytes(fnp.asarray(witness, fnp.uint32).reshape(1))[0]
        return fnp.concatenate([lo4, fnp.zeros(4, fnp.uint8)])

    def _witness_wire(self, witness: Array) -> Array:
        """The witness's wire bytes, framing included: `[OP_BYTES] || len8(8) ||
        nonce_le8`. Split out from `_absorb_witness` so `grind_and_sample` can
        put the same bytes on the stream as part of a draw's framing instead of
        as an absorb of its own."""
        framing = _const_u8(bytes([OP_BYTES]) + _len8(8))
        return fnp.concatenate([framing, self._witness_bytes(witness)])

    def _absorb_witness(self, witness: Array) -> Sha256FieldTranscript:
        return self._absorb(self._witness_wire(witness))

    def grind(
        self, pow_bits: int, *, chunk: int | None = None
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Find a proof-of-work witness — the lowest nonce whose
        `SHA256(state_digest || nonce_le8)` has `pow_bits` leading zero bits —
        and return the transcript advanced past it (the nonce absorbed under the
        `OP_BYTES` wire), plus the witness. Fully traceable
        (`zorch.grind.grind_search` windowed device search); does not raise on
        an exhausted search: `check_witness` is the soundness gate, so which
        witness the search returns is soundness-neutral."""
        witness = self._find_witness(pow_bits, chunk)
        return self._absorb_witness(witness), witness

    def _find_witness(self, pow_bits: int, chunk: int | None) -> Array:
        """The PoW search alone, with nothing absorbed. `grind` puts the witness
        on the wire itself; `grind_and_sample` folds it into a draw's framing."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        if chunk is None:
            chunk = grind_window_for(pow_bits)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        if pow_bits == 0:
            # No work required: the canonical zero witness always passes.
            return fnp.zeros((), fnp.uint32)
        pow_state = self._pow_state()

        def check_batch(counters: Array) -> Array:
            nonce8 = fnp.concatenate(
                [_u32_le_bytes(counters), fnp.zeros((counters.shape[0], 4), fnp.uint8)],
                axis=1,
            )
            return leading_zero_bits_ok(
                sha256_stream_finalize(pow_state, nonce8), pow_bits
            )

        return grind_search(check_batch, 2**32, chunk)

    def grind_and_sample(
        self, pow_bits: int, *, chunk: int | None = None
    ) -> tuple[Sha256FieldTranscript, Array, Array]:
        """Grind, then draw one scalar challenge, as ONE marked region — the
        BLAKE3 row's `grind_and_sample`, on this wire."""
        witness = self._find_witness(pow_bits, chunk)
        t, challenge = self._sample_scalar_after(self._witness_wire(witness))
        return t, witness, challenge

    def check_witness(
        self, witness: Array, *, pow_bits: int
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Verifier mirror of `grind`: check the PoW (`pow_bits == 0` requires
        the canonical witness 0), then absorb the witness REGARDLESS so the
        transcript stays in lockstep. Returns the advanced transcript and the
        device boolean verdict."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        witness = fnp.asarray(witness, fnp.uint32).reshape(())
        if pow_bits == 0:
            ok = witness == fnp.uint32(0)
        else:
            nonce8 = self._witness_bytes(witness)[None, :]
            digs = sha256_stream_finalize(self._pow_state(), nonce8)
            ok = leading_zero_bits_ok(digs, pow_bits)[0]
        return self._absorb_witness(witness), ok

    # ---- element <-> byte serde ----
    def _elem_bytes(self, values: Array) -> Array:
        """Element array -> uint8 `[..., itemsize]` — a direct bitcast to bytes.
        (The wide-binary-field <-> uint8 bitcast once miscompiled on the CPU
        PJRT backend, forcing a uint32-lane detour; that is
        fixed as of the dev20260713 stack.)"""
        return lax.bitcast_convert_type(values, fnp.uint8)

    def _u8_to_elems(self, u8: Array, n: int) -> Array:
        """Flat uint8 `[n * itemsize]` -> `[n]` `dtype` elements (inverse of
        `_elem_bytes`)."""
        return lax.bitcast_convert_type(
            u8.reshape(n, self._item_bytes()), self.dtype
        ).reshape(n)


if TYPE_CHECKING:
    from zorch.transcript import Transcript

    # Seam-conformance pin (docs/reference/conventions.md). Neither field
    # transcript has an in-tree consumer, so without this `Transcript` drift
    # would fail nowhere rather than late.
    _t: type[Transcript] = Sha256FieldTranscript
