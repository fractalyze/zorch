# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fixed-width round buffers (xla#179): cap-width pads/resizes and the
layer-entry donated buffer pool that lays each layer's live prefix in
place."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def _pad_to_width(arr: Array, width: int, neutral: int) -> Array:
    """Extend `arr` to `width` with the fold-neutral fraction tail -- 0 for a
    numerator, 1 for a denominator -- keeping the live prefix at the front. The
    fixed-width round-buffer convention (xla#179), so every round of a phase
    runs at one static shape."""
    pad = width - arr.shape[0]
    if pad == 0:
        return arr
    tail = jnp.zeros((pad,), arr.dtype) if neutral == 0 else jnp.ones((pad,), arr.dtype)
    return jnp.concatenate([arr, tail])


def _resize_zero(arr: Array, width: int) -> Array:
    """Resize a fixed-width round buffer: slice the prefix down or zero-pad up.
    Only correct when the live prefix fits in `width` -- the tail past it is
    dead (masked by the rounds' `live` operand)."""
    if arr.shape[0] >= width:
        return arr[:width]
    return _pad_to_width(arr, width, 0)


# Layer-entry donated cap buffers (xla#179 pad donation). Under a machine
# cap the entry pad materializes FRESH cap-wide buffers every layer -- an
# alloc + zero-fill + prefix copy per plane/eq table, ~2x cap-width writes,
# the top GPU item of the warm decoupled prove (wrapped_concatenate +
# wrapped_broadcast at a cap that is ~19x shard17's live prefix). The pool
# instead holds ONE persistent cap-wide array per (role, width, dtype); each
# layer donates it back to `_lay_prefix`, which writes only the live prefix
# in place. The tail keeps the PREVIOUS layer's bytes: the capped-round
# contract masks every read by the `live` operand and resolves dead slots
# through the sentinel neutral blend, so the tail is never read as data --
# the old pad's zero tail was deterministic filler, not a consumed value.
# Byte-gated by the runner reference test (which reuses the pool across
# layouts, so stale tails are exercised) and the shard golden. Only the
# concrete (non-traced) capped path pools; the traced whole-layer program is
# untouched.
_LAYER_BUF_POOL: dict[tuple[str, int, Any], Array] = {}


@partial(jax.jit, donate_argnums=(0,))
def _lay_prefix(dst: Array, src: Array) -> Array:
    """Write `src` into `dst`'s prefix in place (`dst` donated: the result
    aliases its memory -- no fresh cap-wide buffer, no tail write). Compiled
    once per (cap, natural width, dtype): the same tiny per-natural-width
    executable class as the eager pads it replaces."""
    return jax.lax.dynamic_update_slice(dst, src, (0,))


def _pool_lay(role: str, src: Array, width: int) -> Array:
    """The pooled, donated form of the concrete capped path's
    `_pad_to_width(src, width, 0)` / `_resize_zero` layer-entry lay-in.
    `role` keys the pool entry: two planes share a width/dtype and each must
    own its buffer -- one shared buffer would be donated twice per layer."""
    if src.shape[0] == width:
        return src
    key = (role, width, src.dtype)
    buf = _LAYER_BUF_POOL.get(key)
    if buf is None:
        buf = jnp.zeros((width,), src.dtype)
    out = _lay_prefix(buf, src)
    _LAYER_BUF_POOL[key] = out
    return out


@partial(jax.jit, donate_argnums=(0,))
def _lay_prefix_many(
    dsts: tuple[Array, ...], srcs: tuple[Array, ...]
) -> tuple[Array, ...]:
    """`_lay_prefix` over a batch: one executable lays every (dst, src) pair of
    a layer entry instead of one dispatch per role -- the entry's ~6-7 tiny
    executions collapse to 1. Every dst is donated; the shape combo is static
    per layer layout, so the executable census stays per-layout, same as the
    per-role classes it replaces."""
    return tuple(jax.lax.dynamic_update_slice(d, s, (0,)) for d, s in zip(dsts, srcs))


def _pool_lay_batch(entries: Sequence[tuple[str, Array, int]]) -> list[Array]:
    """The batched `_pool_lay`: lay every `(role, src, width)` entry through
    ONE `_lay_prefix_many` dispatch. Equal-width entries pass through untouched
    (exactly `_pool_lay`'s short-circuit); the rest donate their pooled buffer
    and re-enter the pool as the laid result."""
    out: list[Array] = []
    laid_at: list[int] = []
    bufs: list[Array] = []
    srcs: list[Array] = []
    for role, src, width in entries:
        if src.shape[0] == width:
            out.append(src)
            continue
        buf = _LAYER_BUF_POOL.get((role, width, src.dtype))
        if buf is None:
            buf = jnp.zeros((width,), src.dtype)
        laid_at.append(len(out))
        out.append(buf)  # placeholder, overwritten below
        bufs.append(buf)
        srcs.append(src)
    if bufs:
        laid = _lay_prefix_many(tuple(bufs), tuple(srcs))
        for i, arr in zip(laid_at, laid):
            role, src, width = entries[i]
            _LAYER_BUF_POOL[(role, width, src.dtype)] = arr
            out[i] = arr
    return out
