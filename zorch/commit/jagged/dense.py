# zorch/commit/jagged/dense.py
# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dense packing for the jagged commit.

Variable-height blocks `[h_i, w_i]` are column-major flattened and concatenated
in order (the prefix-sum `t_c` convention P1's `J̃` indicator reads), then
zero-padded to a STATIC `M_max = 2^tier`. The tier is derived host-side from the
total area, exactly as P1's `build_jagged_layout` derives `n_d` — so the device
sees a fixed-shape buffer and the commit compiles once per tier.

"block" is deliberately abstract: how a consumer's trace matrices ("chips") map
to blocks is the consumer's concern, never zorch's (non-negotiable #1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class JaggedLayout:
    """Static shape key + structure metadata for one committable region.

    `log_m` = log2(M_max) (the area tier, the compile-time key); `log_s` =
    log stacking height (S = 2^log_s, K = 2^(log_m - log_s)). `heights`/`widths`
    are per-block counts (host ints) for the structure binding. frozen so it
    hashes by field and can be a `jax.jit` static arg.
    """

    log_m: int
    log_s: int
    heights: tuple[int, ...]
    widths: tuple[int, ...]
    dtype: object

    @property
    def S(self) -> int:
        return 1 << self.log_s

    @property
    def K(self) -> int:
        return 1 << (self.log_m - self.log_s)

    @property
    def m_max(self) -> int:
        return 1 << self.log_m


def from_blocks(
    blocks: Sequence[Array],
    *,
    log_stacking_height: int,
) -> tuple[Array, JaggedLayout]:
    """Pack `blocks` column-major into a static `M_max=2^tier` dense buffer.

    Returns `(packed [M_max], JaggedLayout)`. Tier derived from total area;
    `M_max >= 2^log_stacking_height` so `K >= 1`. All blocks must share a dtype
    (the field dtype); it types the trailing zero padding and rides on the
    layout.
    """
    if not blocks:
        raise ValueError("from_blocks: empty block list")
    if log_stacking_height < 0:
        raise ValueError(f"log_stacking_height must be >= 0, got {log_stacking_height}")
    dtype = blocks[0].dtype
    if any(b.dtype != dtype for b in blocks):
        raise ValueError("from_blocks: every block must share a dtype")
    heights = tuple(int(b.shape[0]) for b in blocks)
    widths = tuple(int(b.shape[1]) for b in blocks)
    total_area = sum(h * w for h, w in zip(heights, widths))

    log_s = log_stacking_height
    log_m = max(log2_ceil_usize(total_area), log_s)
    m_max = 1 << log_m  # >= total_area by construction (log_m >= log2_ceil(area))

    flats = [b.T.reshape(-1) for b in blocks]  # column-major per block
    if m_max > total_area:
        flats.append(jnp.zeros(m_max - total_area, dtype=dtype))
    packed = jnp.concatenate(flats)
    layout = JaggedLayout(
        log_m=log_m, log_s=log_s, heights=heights, widths=widths, dtype=dtype
    )
    return packed, layout
