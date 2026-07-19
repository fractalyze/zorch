# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fixed-width round buffers (xla#179): cap-width pads/resizes and the
layer-entry donated buffer pool that lays each layer's live prefix in
place."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
from frx import Array


def _pad_to_width(arr: Array, width: int, neutral: int) -> Array:
    """Extend `arr` to `width` with the fold-neutral fraction tail -- 0 for a
    numerator, 1 for a denominator -- keeping the live prefix at the front. The
    fixed-width round-buffer convention (xla#179), so every round of a phase
    runs at one static shape."""
    pad = width - arr.shape[0]
    if pad == 0:
        return arr
    tail = fnp.zeros((pad,), arr.dtype) if neutral == 0 else fnp.ones((pad,), arr.dtype)
    return fnp.concatenate([arr, tail])


def _resize_zero(arr: Array, width: int) -> Array:
    """Resize a fixed-width round buffer: slice the prefix down or zero-pad up.
    Only correct when the live prefix fits in `width` -- the tail past it is
    dead (masked by the rounds' `live` operand)."""
    if arr.shape[0] >= width:
        return arr[:width]
    return _pad_to_width(arr, width, 0)


class LayerBuffers:
    """Donated cap-wide layer-entry buffers, owned by ONE prove chain:
    one array per (role, width, dtype), re-donated
    each layer via `_lay_prefix_many` so only the live prefix is written —
    a fresh cap pad per layer costs ~2x cap-width writes, a top GPU item of
    the warm prove. The tail keeps the previous layer's bytes; the capped
    rounds mask every read by the `live` operand, byte-gated by the
    capped-chain test. The one-chain scope is what frees the planes with
    the prove: cap-wide planes run ~5.3 GiB per 80M-cap class, so any
    longer-lived owner caps a resident multi-class prover at one big
    class per 32 GB card."""

    def __init__(self) -> None:
        self.pool: dict[tuple[str, int, Any], Array] = {}


@partial(frx.jit, donate_argnums=(0,))
def _lay_prefix_many(
    dsts: tuple[Array, ...], srcs: tuple[Array, ...]
) -> tuple[Array, ...]:
    """Write each `src` into its donated `dst`'s prefix in place (the result
    aliases the donated buffer -- no fresh cap-wide buffer, no tail write), one
    executable laying every (dst, src) pair of a layer entry instead of one
    dispatch per role. The shape combo is static per layer layout, so the
    executable census stays per-layout."""
    return tuple(
        frx.lax.dynamic_update_slice(d, s, (0,))
        for d, s in zip(dsts, srcs, strict=True)
    )


def _pool_lay_batch(
    entries: Sequence[tuple[str, Array, int]], layer_bufs: LayerBuffers
) -> list[Array]:
    """Lay every `(role, src, width)` entry into its holder's donated
    cap-width buffer through ONE `_lay_prefix_many` dispatch (the batched
    layer-entry lay-in). `role` keys the holder entry: two planes share a
    width/dtype and each must own its buffer -- one shared buffer would be
    donated twice per layer. Equal-width entries pass through untouched; the
    rest donate their held buffer and re-enter the holder as the laid
    result."""
    pool = layer_bufs.pool
    out: list[Array] = []
    laid_at: list[int] = []
    bufs: list[Array] = []
    srcs: list[Array] = []
    for role, src, width in entries:
        if src.shape[0] == width:
            out.append(src)
            continue
        buf = pool.get((role, width, src.dtype))
        if buf is None:
            buf = fnp.zeros((width,), src.dtype)
        laid_at.append(len(out))
        out.append(buf)  # placeholder, overwritten below
        bufs.append(buf)
        srcs.append(src)
    if bufs:
        laid = _lay_prefix_many(tuple(bufs), tuple(srcs))
        for i, arr in zip(laid_at, laid):
            role, src, width = entries[i]
            pool[(role, width, src.dtype)] = arr
            out[i] = arr
    return out
