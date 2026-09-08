# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-element `Transcript` surface over the streaming BLAKE3 core.

The BLAKE3 row of what `sha256_field_transcript.py` is for SHA-256, and shaped
like it: a frozen pytree dataclass whose methods are plain traced state
transitions — jit-first, scan-threadable, no host substrate anywhere. It keeps
the Merlin byte framing (op tag, u64-LE count, squeeze, re-absorb) on the
fixed-shape `Blake3Stream`
([hash-frx's `blake3/streaming.py`](https://github.com/fractalyze/hash-frx/blob/main/hash_frx/blake3/streaming.py)),
so a slice observe / sample is byte-identical to `ByteHashTranscript`'s
`observe_slice` / `sample_slice` over the same hash, and the proof-of-work grind
is `grind`/`check_witness` with `DuplexTranscript`'s exact semantics (device
windowed search via `zorch.grind`; the transcript absorbs the witness
regardless, and `check_witness` is the soundness gate).

Two things are BLAKE3's rather than the framing's, and only one of them is a
choice:

- **The squeeze is an XOF read**, not `byte_transcript`'s
  `HASH(buffer ‖ ctr_le8)` counter chain. Unconditional, no seam.
- **`pow_preimage_bytes`** is the width `state_digest ‖ nonce_le8` is zero-padded
  to before hashing, defaulting to no padding — the pre-image
  `ByteHashTranscript.grind_pow` hashes. Fixed at construction, not per call.

Why each is shaped that way, and the four-row comparison a caller choosing
between transcripts wants, are in `docs/blocks/transcript.md`.

Scheme-agnostic: `dtype` is the challenge element's (scalar) type. `observe`
bitcasts values to bytes and `sample` reinterprets squeezed bytes back, so the
element width is `dtype.itemsize`. The observe/sample surface is one method per
Merlin op tag, for the reason `sha256_field_transcript` states at length: the
tag is transcript-semantic, so it is arity rather than a mode argument.
`byte_transcript` names the wire vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
import numpy as np
from frx import Array, jit, lax
from frx.tree_util import register_dataclass
from hash_frx.blake3 import blake3
from hash_frx.blake3.streaming import Blake3Stream, blake3_stream_init
from hash_frx.fusion import fused_region

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

# BLAKE3's default digest width — the PoW state digest is 32 B, and it is the
# width `_validate_pow_bits` bounds a difficulty against.
_DIGEST_BYTES = 32
# The unpadded PoW pre-image: `state_digest ‖ nonce_le8`, as on the byte
# transcript. `pow_preimage_bytes` zero-pads beyond it.
_POW_PREIMAGE_BYTES = _DIGEST_BYTES + 8
# Hash mode, hoisted so it is not rebuilt per trace. Only the wide-pre-image arm
# of `_pow_digests` passes it — the marked entry is hash-mode by construction.
_MODE = blake3.hash_mode()


def _const_u8(data: bytes) -> Array:
    """A compile-time-constant byte payload as a device uint8 array."""
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _nonce8(counters: Array) -> Array:
    """uint32 `[B]` nonces on the u64-LE wire: uint8 `[B, 8]`.

    The high four bytes are zero because the search domain is uint32, as in
    `DuplexTranscript._grind_search`. The bitcast is `_elem_bytes`' and the
    SHA-256 row's spelling of the same conversion — one wire, one spelling."""
    return fnp.concatenate(
        [
            lax.bitcast_convert_type(counters, fnp.uint8),
            fnp.zeros((counters.shape[0], 4), fnp.uint8),
        ],
        axis=1,
    )


# ============================================================================
# The squeeze marker, mirroring `zorch.sha256_squeeze`.
#
# A squeeze is absorb -> finalize -> absorb, and each of those is its own jit
# zone entered through the resumable state, so unmarked it lowers to a long
# chain of tiny launches. The SHA-256 row measured ~14 launches per
# `sample_scalar` and notes that on a latency-bound prove the cost is the launch
# count, not the arithmetic. BLAKE3 had no marker at all, and it shows: a
# sequential `sample_scalar` chain measures 16.6x the SHA-256 transcript
# (150.2 ms vs 9.0 ms for 256 squeezes on Metal), which is worth ~177 ms of a
# 429 ms m=28 prove on the blake3 arm — the arm the flock-challenge harness
# uses. See flock-zorch#298.
#
# hash-frx's own marker covers a whole message, which a squeeze is not: it is
# entered mid-stream at a runtime position. So the state rides as OPERANDS (the
# seven `Blake3Stream` leaves) and only the squeeze width is an attr, which
# keeps one kernel serving every stream position instead of recompiling per
# `block_len`. With no emitter installed the decomposition inlines and is
# byte-identical, so this is safe to land before the kernel exists.
# ============================================================================
BLAKE3_SQUEEZE_MARKER = "zorch.blake3_squeeze"
BLAKE3_SQUEEZE_MARKER_VERSION = 1

# The absorb half of the same story. Once the squeeze is fused, `observe` is
# what is left: it goes through the plain streaming absorb, and at n=256 rounds
# it measures 44.2 ms against the SHA-256 transcript's 0.36 ms — 123x, and the
# whole residual FS deficit on the blake3 arm.
#
# `length` is an attr, as `nbytes` is for the squeeze. That is affordable here
# because a real m=28 prove uses only twelve distinct payload lengths across 90
# absorbs, and 79 of those are 17 or 18 bytes — so a handful of kernel variants
# serve everything, and the hot ones are under a block.
BLAKE3_ABSORB_MARKER = "zorch.blake3_absorb"
BLAKE3_ABSORB_MARKER_VERSION = 1

# The third of the streaming trio, and the one the first two left behind. A
# squeeze's finalize rides inside `zorch.blake3_squeeze`, but the proof-of-work
# grind opens with a bare `finalize` — a root compression with no re-absorb to
# sit beside — so it fell outside every marker. On Metal that made it the
# largest unmarked scope in `open`: 683 dispatches, 17.0% of the phase, every
# one a single threadgroup. See flock-zorch#308.
#
# Same ABI as the other two: the state rides as operands and the output width is
# the attr, so one kernel serves every stream position.
BLAKE3_FINALIZE_MARKER = "zorch.blake3_finalize"
BLAKE3_FINALIZE_MARKER_VERSION = 1


def _blake3_absorb_region(
    state: Blake3Stream, payload: Array, **_attrs: object
) -> Blake3Stream:
    """The `zorch.blake3_absorb` decomposition, entered at the runtime stream
    position carried in the state leaves. `_attrs` (`length`) is composite
    metadata the emitter parses; the decomposition needs only the operands.

    `state` rides as ONE pytree operand: `Blake3Stream` is a registered
    dataclass, so `lax.composite` flattens it to its seven leaves in field
    order, which IS the marker's operand ABI. Spelling the leaves out by hand
    would restate that order at every call site, where a field added or moved
    upstream lands as silent state corruption rather than an error."""
    return state.absorb(payload)


@partial(jit, inline=True)
def _blake3_absorb_zone(state: Blake3Stream, payload: Array) -> Blake3Stream:
    """One marked absorb. The payload length is static at every call site (it
    comes from a framing constant plus a fixed-width serialization), so it rides
    as an attr and the emitter gets a compile-time block count.

    `inline=True` keeps a call site already inside an outer jit byte-identical,
    mirroring `_blake3_squeeze_zone`."""
    length = payload.shape[0]
    if length == 0:
        return state
    return fused_region(
        _blake3_absorb_region,
        state,
        payload,
        name=BLAKE3_ABSORB_MARKER,
        version=BLAKE3_ABSORB_MARKER_VERSION,
        length=length,
    )


def _blake3_finalize_region(state: Blake3Stream, nbytes: int) -> Array:
    """The `zorch.blake3_finalize` decomposition: read `nbytes` of extendable
    output at the runtime stream position carried in the state leaves, without
    advancing it. `composite` passes attrs as keywords, so `nbytes` binds from
    the marker.

    `state` rides as ONE pytree operand, for the reason `_blake3_absorb_region`
    gives."""
    return state.finalize(nbytes)


@partial(jit, static_argnames=("nbytes",), inline=True)
def _blake3_finalize_zone(state: Blake3Stream, nbytes: int) -> Array:
    """One marked finalize, for the readers that take a digest without putting
    it back on the stream — which is a squeeze minus its two absorbs, so the
    emitter has strictly less to do than `zorch.blake3_squeeze`'s.

    `inline=True` keeps a call site already inside an outer jit byte-identical,
    mirroring `_blake3_absorb_zone`."""
    return fused_region(
        _blake3_finalize_region,
        state,
        name=BLAKE3_FINALIZE_MARKER,
        version=BLAKE3_FINALIZE_MARKER_VERSION,
        nbytes=nbytes,
    )


def _blake3_squeeze_hop(
    state: Blake3Stream, framing: Array, nbytes: int
) -> tuple[Blake3Stream, Array]:
    """The plain hop: absorb `framing`, read `nbytes` of extendable output,
    re-absorb it. Returns the advanced state and the raw squeezed bytes;
    reinterpreting those to field elements stays with the caller, so the region
    carries no dtype."""
    absorbed = state.absorb(framing)
    squeezed = absorbed.finalize(nbytes)
    return absorbed.absorb(squeezed), squeezed


@partial(jit, static_argnames=("nbytes",), inline=True)
def _blake3_squeeze_zone(
    state: Blake3Stream, framing: Array, nbytes: int
) -> tuple[Blake3Stream, Array]:
    """The marked hop as ONE dispatch rather than three, so an eager draw fires
    a single fused FS kernel instead of three host dispatches and three syncs.

    `inline=True` keeps a call site already inside an outer jit byte-identical,
    mirroring `_sha256_squeeze_zone`. (`grind`'s pre-image is a whole message,
    so it takes hash-frx's own marker instead — see `_pow_digests`.)

    `_blake3_squeeze_hop` IS the decomposition — `composite` passes attrs as
    keywords, so its `nbytes` binds from the marker. `state` rides as one pytree
    operand, for the reason `_blake3_absorb_region` gives.

    No zero-width guard, unlike `_sha256_squeeze_zone`: `_squeeze` is the only
    caller and it returns before reaching here when `nbytes == 0`, because the
    XOF read refuses a zero length -- so the guard would be unreachable, and
    would raise anyway if it were reached."""
    return fused_region(
        _blake3_squeeze_hop,
        state,
        framing,
        name=BLAKE3_SQUEEZE_MARKER,
        version=BLAKE3_SQUEEZE_MARKER_VERSION,
        nbytes=nbytes,
    )


@partial(
    register_dataclass,
    data_fields=["state"],
    meta_fields=["dtype", "pow_preimage_bytes"],
)
@dataclass(frozen=True)
class Blake3FieldTranscript:
    """Device BLAKE3 transcript satisfying `transcript.Transcript`, threadable
    through a `lax.scan` / `@jit` like `DuplexTranscript`. `state` is the only
    data field, so the pytree structure is fixed however much has been absorbed
    — the property that lets a round loop carry it. `dtype` (static) is the
    challenge element type; `pow_preimage_bytes` (static) is the PoW wire."""

    state: Blake3Stream
    dtype: Any
    pow_preimage_bytes: int = _POW_PREIMAGE_BYTES

    @property
    def field(self) -> Any:
        return self.dtype

    @property
    def has_dedicated_fusion(self) -> bool:
        # False where the SHA-256 row reads True: this flag is about the FS hop,
        # whose compressions are entered through the resumable state rather than
        # through hash-frx's marked whole-message region, and hash-frx's own
        # BLAKE3 rows report False regardless. Consumers take their plain
        # decomposition paths. `zorch.blake3_{absorb,squeeze,finalize}` do not
        # change this: they are zorch's own regions over the resumable state,
        # not the hash-frx region this flag names. `_pow_digests` is the
        # one path that does reach the marked region — it hashes a whole
        # message — and it is not an FS hop.
        return False

    @classmethod
    def new(
        cls,
        domain: bytes,
        dtype: Any,
        *,
        pow_preimage_bytes: int = _POW_PREIMAGE_BYTES,
    ) -> Blake3FieldTranscript:
        # Validated here rather than in `__post_init__`: the width is fixed at
        # construction, and `__post_init__` would re-check it on every `replace`
        # the framing does and on every pytree unflatten.
        if pow_preimage_bytes < _POW_PREIMAGE_BYTES:
            raise ValueError(
                f"pow_preimage_bytes must be at least {_POW_PREIMAGE_BYTES} — "
                f"the nonce does not fit below it — got {pow_preimage_bytes}"
            )
        seed = _const_u8(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))
        return cls(
            blake3_stream_init().absorb(seed), np.dtype(dtype), pow_preimage_bytes
        )

    def _item_bytes(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    def _absorb(self, payload: Array) -> Blake3FieldTranscript:
        return replace(self, state=_blake3_absorb_zone(self.state, payload))

    def observe(self, values: Array) -> Blake3FieldTranscript:
        """Absorb `values` under slice framing: `[OP_OBSERVE, KIND_SLICE] ||
        len8(count) || serialized bytes`. Byte-identical to the byte
        transcript's `observe_slice` of the same serialized bytes."""
        vals_u8 = self._elem_bytes(values).reshape(-1)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(fnp.concatenate([framing, vals_u8]))

    def observe_scalar(self, value: Array) -> Blake3FieldTranscript:
        """Absorb under scalar framing `[OP_OBSERVE, KIND_SCALAR] || elem_bytes`
        — no length prefix, a scalar's width being implicit in the dtype. A 0-d
        `value` is one op; an `[n]` array is n ops (one per element, in order),
        built as ONE absorb payload. Distinct from `observe` (the KIND tag
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
    ) -> tuple[Blake3FieldTranscript, Array]:
        """Put `payload` on the stream and draw a scalar, as ONE marked region.

        This is the merge every fused pair on this row is made of. Absorb is a
        stream, so `absorb(P); squeeze(F)` and `squeeze(P || F)` leave the same
        state, and `_squeeze` already absorbs its framing before reading — so a
        payload that would have been its own absorb can ride the draw instead.
        Byte-identical by construction, and one marked region rather than two.
        """
        framing = fnp.concatenate(
            [payload, _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))]
        )
        t, squeezed = self._squeeze(framing, self._item_bytes())
        return t, t._u8_to_elems(squeezed, 1)[0]

    def observe_scalar_and_sample(
        self, value: Array
    ) -> tuple[Blake3FieldTranscript, Array]:
        """`observe_scalar` then `sample_scalar`, as one marked region."""
        return self._sample_scalar_after(self._scalar_observe_wire(value))

    def observe_label(self, label: bytes) -> Blake3FieldTranscript:
        """Absorb a domain-separation label `[OP_LABEL] || len8(len) || label`.
        A compile-time host constant (labels are literals), so the whole absorb
        is one constant payload."""
        return self._absorb(
            _const_u8(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))
        )

    def observe_bytes(self, data: Array) -> Blake3FieldTranscript:
        """Absorb opaque bytes (e.g. a Merkle root computed on-device) under
        `[OP_BYTES] || len8(len) || data`. `data` is a uint8 array whose length
        is static (it rides the framing prefix)."""
        data = fnp.asarray(data, fnp.uint8).reshape(-1)
        framing = _const_u8(bytes([OP_BYTES]) + _len8(int(data.shape[0])))
        return self._absorb(fnp.concatenate([framing, data]))

    def _squeeze(
        self, framing: Array, nbytes: int
    ) -> tuple[Blake3FieldTranscript, Array]:
        """Absorb `framing`, read `nbytes` of extendable output, re-absorb it.

        Finalizing does not end the stream, so the state a caller gets back is
        the absorbed one and nothing is spent by squeezing.
        """
        if nbytes == 0:
            # A zero-width draw still frames, matching the byte transcript's
            # empty squeeze; the XOF read itself refuses a zero length.
            return self._absorb(framing), fnp.zeros((0,), fnp.uint8)
        state, squeezed = _blake3_squeeze_zone(self.state, framing, nbytes)
        return replace(self, state=state), squeezed

    def sample(self, n: int = 1) -> tuple[Blake3FieldTranscript, Array]:
        """Squeeze `n` challenge elements: absorb `[OP_SQUEEZE, KIND_SLICE] ||
        len8(n)`, read `n * itemsize` bytes of extendable output, re-absorb them,
        and reinterpret to `n` elements of `dtype`."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(n))
        t, squeezed = self._squeeze(framing, n * self._item_bytes())
        return t, t._u8_to_elems(squeezed, n)

    def sample_scalar(self) -> tuple[Blake3FieldTranscript, Array]:
        """Squeeze one challenge under scalar framing: absorb `[OP_SQUEEZE,
        KIND_SCALAR]`, read `itemsize` bytes, re-absorb, reinterpret to one
        `dtype` element (0-D). Distinct from `sample(1)` (the KIND tag
        differs)."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))
        t, squeezed = self._squeeze(framing, self._item_bytes())
        return t, t._u8_to_elems(squeezed, 1)[0]

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Blake3FieldTranscript, Array]:
        return self.observe(values).sample(n)

    # ---- proof-of-work (DuplexTranscript's grind/check_witness shape) ----
    def _state_digest(self) -> Array:
        """`BLAKE3(buffer)` at the digest width — the first 32 bytes of every PoW
        pre-image, matching the byte transcript's `HASH(state_digest ||
        nonce_le8)`. The same finalize a squeeze reads, so the grind opens with
        no digest construction of its own — but a squeeze's finalize is inside
        `zorch.blake3_squeeze` and this one has no re-absorb to sit beside, so it
        takes `zorch.blake3_finalize` of its own (flock-zorch#308)."""
        return _blake3_finalize_zone(self.state, _DIGEST_BYTES)

    def _pow_digests(self, state_digest: Array, counters: Array) -> Array:
        """PoW candidate digests for a uint32 `[B]` counter batch: uint8
        `[B, 32]`.

        The rows are the wire spelled out: `state_digest ‖ nonce_le8` zero-padded
        to `pow_preimage_bytes`, hashed as a whole message. Hashing it rather
        than assembling a compression is what lets the width be a free parameter
        — the pre-image is only one compression while it fits in a block, and
        `unmarked_hash` keeps being right past that without this knowing BLAKE3's
        flag schedule.

        **Marked while the pre-image is that one compression.** `xof` carries
        the `hash_frx.blake3` marker a recognizing emitter collapses into a
        single kernel — the fusion north star (`docs/README.md`) applied to a
        hash permutation — and it traces cheaper besides, `tree_hash` being a
        value-keyed jit zone where `unmarked_hash` re-traces its body per call
        site. Past a block the unmarked arm takes over, because a marked call
        compiles that whole unrolled body, "affordable at a block and not at a
        chunk" in hash-frx's terms, and the width is the caller's knob. The
        guard reads bytes rather than compressions because `_DIGEST_BYTES` keeps
        the root's own output inside one block too.

        Both arms hash the same bytes; only the lowering differs. The saving is
        per LAUNCH rather than per byte — ~23 launches a window at the default —
        so a narrower `GRIND_WINDOW` multiplies it. Re-measure if that moves.
        """
        batch = counters.shape[0]
        rows = fnp.concatenate(
            [
                fnp.broadcast_to(state_digest, (batch, _DIGEST_BYTES)),
                _nonce8(counters),
                fnp.zeros(
                    (batch, self.pow_preimage_bytes - _POW_PREIMAGE_BYTES), fnp.uint8
                ),
            ],
            axis=1,
        )
        if self.pow_preimage_bytes > blake3.BLOCK_LEN:
            return blake3.unmarked_hash(rows, _MODE, _DIGEST_BYTES)
        return blake3.xof(rows, _DIGEST_BYTES)

    def _witness_wire(self, witness: Array) -> Array:
        """The witness's wire bytes, framing included: `[OP_BYTES] || len8(8) ||
        nonce_le8`. Split out from `_absorb_witness` so `grind_and_sample` can
        put the same bytes on the stream as part of a draw's framing instead of
        as an absorb of its own."""
        nonce8 = _nonce8(fnp.asarray(witness, fnp.uint32).reshape(1))[0]
        return fnp.concatenate([_const_u8(bytes([OP_BYTES]) + _len8(8)), nonce8])

    def _absorb_witness(self, witness: Array) -> Blake3FieldTranscript:
        return self._absorb(self._witness_wire(witness))

    def grind(
        self, pow_bits: int, *, chunk: int | None = None
    ) -> tuple[Blake3FieldTranscript, Array]:
        """Find a proof-of-work witness — the lowest nonce whose pre-image digest
        has `pow_bits` leading zero bits — and return the transcript advanced
        past it (the nonce absorbed under the `OP_BYTES` wire), plus the witness.
        Fully traceable (`zorch.grind.grind_search` windowed device search); does
        not raise on an exhausted search: `check_witness` is the soundness gate,
        so which witness the search returns is soundness-neutral."""
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
        state_digest = self._state_digest()

        def check_batch(counters: Array) -> Array:
            return leading_zero_bits_ok(
                self._pow_digests(state_digest, counters), pow_bits
            )

        return grind_search(check_batch, 2**32, chunk)

    def grind_and_sample(
        self, pow_bits: int, *, chunk: int | None = None
    ) -> tuple[Blake3FieldTranscript, Array, Array]:
        """Grind, then draw one scalar challenge, as ONE marked region — the
        witness rides the draw's framing (see `_sample_scalar_after`). Byte-
        identical to `grind(...)` followed by `sample_scalar()`."""
        witness = self._find_witness(pow_bits, chunk)
        t, challenge = self._sample_scalar_after(self._witness_wire(witness))
        return t, witness, challenge

    def check_witness(
        self, witness: Array, *, pow_bits: int
    ) -> tuple[Blake3FieldTranscript, Array]:
        """Verifier mirror of `grind`: check the PoW (`pow_bits == 0` requires
        the canonical witness 0), then absorb the witness REGARDLESS so the
        transcript stays in lockstep. Returns the advanced transcript and the
        device boolean verdict."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        witness = fnp.asarray(witness, fnp.uint32).reshape(())
        if pow_bits == 0:
            ok = witness == fnp.uint32(0)
        else:
            digests = self._pow_digests(self._state_digest(), witness.reshape(1))
            ok = leading_zero_bits_ok(digests, pow_bits)[0]
        return self._absorb_witness(witness), ok

    # ---- element <-> byte serde ----
    def _elem_bytes(self, values: Array) -> Array:
        """Element array -> uint8 `[..., itemsize]` — a direct bitcast to bytes."""
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
    _t: type[Transcript] = Blake3FieldTranscript
