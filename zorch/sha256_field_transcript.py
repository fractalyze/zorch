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
    OP_DOMAIN,
    OP_OBSERVE,
    OP_SQUEEZE,
)
from zorch.hash.sha256 import (
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)


def _len8(n: int) -> bytes:
    """A length / count as 8 little-endian bytes (the transcript's only integer
    encoding — fixint u64-LE everywhere)."""
    return int(n).to_bytes(8, "little")


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

    def observe_scalar(self, value: Array) -> Sha256FieldTranscript:
        """Absorb one element under scalar framing: `[OP_OBSERVE, KIND_SCALAR] ||
        lo‖hi bytes` — no length prefix (a scalar's width is implicit in the
        consumer). Byte-identical to `ByteHashTranscript.observe_scalar`; the
        per-element framing a byte challenger (flock's `observe_f128`) uses,
        where `observe`'s count-prefixed slice framing would not match."""
        vals_u8 = lax.bitcast_convert_type(value, jnp.uint8).reshape(-1)
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SCALAR]))
        return self._absorb(jnp.concatenate([framing, vals_u8]))

    def sample_scalar(
        self, nbytes: int | None = None
    ) -> tuple[Sha256FieldTranscript, Array]:
        """Squeeze one challenge under scalar framing: absorb `[OP_SQUEEZE,
        KIND_SCALAR]` (no count), counter-squeeze `nbytes` (default one `dtype`
        element), re-absorb, reinterpret to `dtype`. Byte-identical to
        `ByteHashTranscript.sample_scalar` — the framing `sample_f128` uses."""
        nbytes = self._item_bytes() if nbytes is None else nbytes
        t = self._absorb(_const_u8(bytes([OP_SQUEEZE, KIND_SCALAR])))
        squeezed = t._squeeze_bytes(nbytes)
        t = t._absorb(squeezed)
        return t, squeezed.view(self.dtype)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[Sha256FieldTranscript, Array]:
        return self.observe(values).sample(n)
