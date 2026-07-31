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

import frx.numpy as fnp
import numpy as np
from frx import Array, jit, lax
from frx.tree_util import register_dataclass

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
from zorch.fusion import fused_region
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
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _u32_le_bytes(values: Array) -> Array:
    """uint32 `[...]` -> uint8 `[..., 4]`, little-endian."""
    return lax.bitcast_convert_type(values, fnp.uint8)


# ============================================================================
# Squeeze-hop fusion marker. `zorch.sha256` marks the COMPRESSION; this marks
# the whole squeeze around it — the same layering `zorch.duplex_fs` adds over
# `zorch.poseidon2`, for the same reason.
#
# A squeeze is `absorb(framing) -> counter-squeeze -> re-absorb`. The streaming
# state is branchless, so each absorb compresses speculatively and selects and
# finalize emits both padding candidates: four `zorch.sha256` regions, each a
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


# ============================================================================
# `zorch.sample_distinct` — one rejection-sampling DRAW as one kernel.
#
# Rejection sampling is irreducibly serial: each candidate is a squeeze and the
# squeezes are a Fiat-Shamir chain, so the only thing left to win is what a
# draw costs in kernels. Marking the squeeze alone leaves the bookkeeping
# around it — the membership test over the accepted prefix, the append, the
# accept/reject select — as three more launches, and on an idle card those cost
# as much as the squeeze itself: the body is latency-bound, so op COUNT is the
# cost and rewriting any single op into a cheaper one buys nothing.
#
# The framing rides as an OPERAND and the limb width as an attr, which is what
# lets one region serve both conventions in use — a scalar-framed draw reducing
# the low uint64 limb and a slice-framed one reducing the low uint32 — without
# either being baked into the emitter.
#
# The decomposition is the plain draw, so with no emitter the marker inlines
# byte-identically, exactly like the squeeze hop above.
# ============================================================================
SAMPLE_DISTINCT_MARKER = "zorch.sample_distinct"
SAMPLE_DISTINCT_MARKER_VERSION = 1


# Limb widths a draw may reduce by, and the unsigned dtype each bitcasts to.
# Restricted to native widths on purpose: assembling the integer from bytes
# instead costs a shift/or chain that does NOT fold away inside the marked
# region, and measured 22% slower per draw than the bitcast it replaced.
_LIMB_DTYPES = {1: fnp.uint8, 2: fnp.uint16, 4: fnp.uint32, 8: fnp.uint64}


def _le_limb(data: Array, limb_bytes: int) -> Array:
    """`data[:limb_bytes]` read as a little-endian unsigned integer — the same
    bitcast the unmarked samplers do, so the marked draw reduces identically."""
    return lax.bitcast_convert_type(
        data[:limb_bytes], _LIMB_DTYPES[limb_bytes]
    ).reshape(())


def _distinct_draw(
    state: Sha256State,
    framing: Array,
    out: Array,
    n: Array,
    *,
    nbytes: int,
    block_len: int,
    limb_bytes: int,
) -> tuple[Sha256State, Array, Array]:
    """One draw: squeeze, reduce the low `limb_bytes` limb mod `block_len`, and
    append the position to `out[:n]` unless it is already there. `out` keeps its
    full width throughout — only `n` says how much of it is live.

    The squeeze is the MARKED hop, so this region nests `zorch.sha256_squeeze`
    (as `zorch.duplex_fs` nests `zorch.poseidon2`). That nesting is what keeps
    the marker free before its own emitter exists: dropping to the plain
    `_squeeze_hop` here would un-fuse the squeeze and blow the draw from 7
    launch-shaped ops back out to 17."""
    state, squeezed = _sha256_squeeze_zone(state, framing, nbytes)
    limb = _le_limb(squeezed, limb_bytes)
    pos = (limb % limb.dtype.type(block_len)).astype(fnp.int32)
    idx = fnp.arange(out.shape[0], dtype=fnp.int32)
    hit = fnp.any((idx < n) & (out == pos))
    return (
        state,
        fnp.where(hit, out, out.at[n].set(pos)),
        fnp.where(hit, n, n + fnp.int32(1)),
    )


def _sample_distinct_region(
    h: Array,
    pending: Array,
    counts: Array,
    framing: Array,
    out: Array,
    n: Array,
    *,
    nbytes: int,
    block_len: int,
    limb_bytes: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """The `zorch.sample_distinct` decomposition. The stream position rides in
    the state leaves and the accepted count in `n`, both operands, so one
    recompile-free kernel serves every draw of every level."""
    state, out, n = _distinct_draw(
        Sha256State(h, pending, counts),
        framing,
        out,
        n,
        nbytes=nbytes,
        block_len=block_len,
        limb_bytes=limb_bytes,
    )
    return state.h, state.pending, state.counts, out, n


@partial(jit, static_argnames=("nbytes", "block_len", "limb_bytes"), inline=True)
def _sample_distinct_zone(
    state: Sha256State,
    framing: Array,
    out: Array,
    n: Array,
    *,
    nbytes: int,
    block_len: int,
    limb_bytes: int,
) -> tuple[Sha256State, Array, Array]:
    """The marked draw as one compiled dispatch carrying the
    `zorch.sample_distinct` composite. `inline=True` keeps a call site already
    inside an outer jit byte-identical (mirrors `_sha256_squeeze_zone`)."""
    h, pending, counts, out, n = fused_region(
        _sample_distinct_region,
        state.h,
        state.pending,
        state.counts,
        framing,
        out,
        n,
        name=SAMPLE_DISTINCT_MARKER,
        version=SAMPLE_DISTINCT_MARKER_VERSION,
        nbytes=nbytes,
        block_len=block_len,
        limb_bytes=limb_bytes,
    )
    return Sha256State(h, pending, counts), out, n


def _leading_zero_bits_ok(digests: Array, bits: int) -> Array:
    """Whether each digest (uint8 `[B, 32]`) has >= `bits` leading zero bits,
    big-endian (digest[..., 0] most significant). Traceable; byte-identical to
    `byte_transcript._leading_zero_bits_ok`."""
    full, extra = divmod(bits, 8)
    ok = fnp.all(digests[:, :full] == 0, axis=1)
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
    def field(self) -> Any:
        return self.dtype

    @property
    def has_dedicated_fusion(self) -> bool:
        # The COMPRESSION lowers to a GPU kernel via the zorch.sha256 marker.
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
        vals_u8 = self._elem_bytes(value).reshape(-1, self._item_bytes())
        framing = fnp.broadcast_to(
            _const_u8(bytes([OP_OBSERVE, KIND_SCALAR])), (vals_u8.shape[0], 2)
        )
        return self._absorb(fnp.concatenate([framing, vals_u8], axis=1).reshape(-1))

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

    # ---- rejection-sampled query positions (`zorch.sample_distinct`) ----
    def _sample_distinct(
        self, framing: Array, block_len: int, count: int, limb_bytes: int
    ) -> tuple[Sha256FieldTranscript, Array]:
        """`count` DISTINCT positions in `[0, block_len)`, sorted ascending: one
        marked draw per candidate, re-drawing on a repeat. A device `while_loop`
        — `jit`-safe, never leaves the device — whose body is one marked hop, so
        a vendor that emits the marker pays one kernel per candidate."""
        if count > block_len:
            raise ValueError(
                f"cannot sample {count} distinct positions from a block of "
                f"{block_len}"
            )
        nbytes = self._item_bytes()
        # An over-wide limb would slice past the squeezed element, which a
        # traced slice CLAMPS rather than raises on — a silently wrong position,
        # so reject it here alongside the non-native widths.
        if limb_bytes not in _LIMB_DTYPES or limb_bytes > nbytes:
            raise ValueError(
                f"limb_bytes must be one of {sorted(_LIMB_DTYPES)} and at most "
                f"the {nbytes}-byte element width, got {limb_bytes}"
            )

        def body(
            carry: tuple[Sha256State, Array, Array]
        ) -> tuple[Sha256State, Array, Array]:
            state, out, n = carry
            return _sample_distinct_zone(
                state,
                framing,
                out,
                n,
                nbytes=nbytes,
                block_len=block_len,
                limb_bytes=limb_bytes,
            )

        state, out, _ = lax.while_loop(
            lambda c: c[2] < count,
            body,
            (self.state, fnp.zeros(count, fnp.int32), fnp.int32(0)),
        )
        return replace(self, state=state), fnp.sort(out)

    def sample_distinct(
        self, block_len: int, count: int, *, limb_bytes: int = 4
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Rejection-sample distinct positions drawing under SLICE framing —
        each candidate is a `sample(1)`. Separate from `sample_distinct_scalar`
        for the same reason `sample` and `sample_scalar` are separate: the KIND
        tag is transcript-semantic, so the two draw different challenge streams
        and a mode flag would hide that in an argument."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(1))
        return self._sample_distinct(framing, block_len, count, limb_bytes)

    def sample_distinct_scalar(
        self, block_len: int, count: int, *, limb_bytes: int = 4
    ) -> tuple[Sha256FieldTranscript, Array]:
        """`sample_distinct` drawing under SCALAR framing — each candidate is a
        `sample_scalar()`."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))
        return self._sample_distinct(framing, block_len, count, limb_bytes)

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

    def _absorb_witness(self, witness: Array) -> Sha256FieldTranscript:
        framing = _const_u8(bytes([OP_BYTES]) + _len8(8))
        return self._absorb(fnp.concatenate([framing, self._witness_bytes(witness)]))

    def grind(
        self, pow_bits: int, *, chunk: int = GRIND_WINDOW
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Find a proof-of-work witness — the lowest nonce whose
        `SHA256(state_digest || nonce_le8)` has `pow_bits` leading zero bits —
        and return the transcript advanced past it (the nonce absorbed under the
        `OP_BYTES` wire), plus the witness. Fully traceable
        (`zorch.grind.grind_search` windowed device search); does not raise on
        an exhausted search: `check_witness` is the soundness gate, so which
        witness the search returns is soundness-neutral."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        if pow_bits == 0:
            # No work required: the canonical zero witness always passes.
            witness = fnp.zeros((), fnp.uint32)
            return self._absorb_witness(witness), witness
        pow_state = self._pow_state()

        def check_batch(counters: Array) -> Array:
            nonce8 = fnp.concatenate(
                [_u32_le_bytes(counters), fnp.zeros((counters.shape[0], 4), fnp.uint8)],
                axis=1,
            )
            return _leading_zero_bits_ok(
                sha256_stream_finalize(pow_state, nonce8), pow_bits
            )

        witness = grind_search(check_batch, 2**32, chunk)
        return self._absorb_witness(witness), witness

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
            ok = _leading_zero_bits_ok(digs, pow_bits)[0]
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
