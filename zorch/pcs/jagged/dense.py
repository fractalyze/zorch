# zorch/pcs/jagged/dense.py
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


def log_area_tier(total_area: int) -> int:
    """Log-area tier shared by the dense buffer (`log_m`) and the jagged
    indicator (`n_d`). The +1 keeps `t_L < 2^tier` strict (a power-of-two area
    would otherwise need t_L == 2^tier), so prefix sums always fit the bit
    width. One definition site: the opening seam requires `log_m == n_d`, so
    both sides must derive the tier identically."""
    return log2_ceil_usize(total_area) + 1


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


# Largest permitted log-area tier: 2^30 stays below every supported field
# prime (counts embed canonically into the structure hash via `_structure_vec`,
# so a count >= the prime would alias a smaller one), and keeps the int32
# prefix-sum decode (`_decode_prefix_sums`, `partial_eval`'s searchsorted
# offsets) overflow-free.
_MAX_AREA_TIER = 30


def validate_layout(layout: JaggedLayout) -> None:
    """Fail loud on a structurally invalid `JaggedLayout`.

    The verifier consumes the layout from an untrusted statement, so every
    count — including the block count itself, the structure hash's first
    element — is validated before any arithmetic reads it. Full rationale:
    docs/jagged.md § Verifier input contract.
    """
    if len(layout.heights) != len(layout.widths):
        raise ValueError(
            f"layout heights/widths must pair up, got {len(layout.heights)} "
            f"heights vs {len(layout.widths)} widths"
        )
    if not layout.heights:
        raise ValueError("layout must declare at least one block")
    for label, counts in (("height", layout.heights), ("width", layout.widths)):
        for c in counts:
            if c < 0:
                raise ValueError(f"negative block {label} {c}")
    total_area = sum(h * w for h, w in zip(layout.heights, layout.widths))
    if total_area == 0:
        raise ValueError("layout area must be positive")
    n_d = log_area_tier(total_area)
    if n_d > _MAX_AREA_TIER:
        raise ValueError(f"log-area tier {n_d} exceeds the safe bound {_MAX_AREA_TIER}")
    # The area bounds h·w products only; a zero partner hides an oversized
    # count from it (e.g. width 0 with a huge height), so bound each count
    # individually by the tier capacity. Same for the block count: zero-area
    # blocks grow it without growing the area.
    cap = 1 << n_d
    if len(layout.heights) > cap:
        raise ValueError(
            f"block count {len(layout.heights)} exceeds the tier capacity 2^{n_d}"
        )
    for label, counts in (("height", layout.heights), ("width", layout.widths)):
        for c in counts:
            if c > cap:
                raise ValueError(f"block {label} {c} exceeds the tier capacity 2^{n_d}")
    # log_m is bounded but NOT pinned to n_d: a stacking height above the area
    # tier legitimately dominates it (`from_blocks` takes max(tier, log_s));
    # the log_m == n_d equality is an open/verify concern, not validation's.
    if layout.log_m > _MAX_AREA_TIER:
        raise ValueError(
            f"log_m {layout.log_m} exceeds the safe bound {_MAX_AREA_TIER}"
        )
    if not 0 <= layout.log_s <= layout.log_m:
        raise ValueError(
            f"log_s must satisfy 0 <= log_s <= log_m, got log_s={layout.log_s}, "
            f"log_m={layout.log_m}"
        )


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
    log_m = max(log_area_tier(total_area), log_s)
    # Guard before the 2^log_m allocation below — validate_layout runs only
    # later (commit's _indicator_inputs), after the buffer would already exist.
    if log_m > _MAX_AREA_TIER:
        raise ValueError(f"log_m {log_m} exceeds the safe bound {_MAX_AREA_TIER}")
    m_max = 1 << log_m  # > total_area by construction (log_area_tier is strict)

    flats = [b.T.reshape(-1) for b in blocks]  # column-major per block
    if m_max > total_area:
        flats.append(jnp.zeros(m_max - total_area, dtype=dtype))
    packed = jnp.concatenate(flats)
    layout = JaggedLayout(
        log_m=log_m, log_s=log_s, heights=heights, widths=widths, dtype=dtype
    )
    return packed, layout
