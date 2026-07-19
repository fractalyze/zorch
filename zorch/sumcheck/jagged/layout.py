# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic segment-layout recurrences for jagged sumcheck.

The numpy gather that remaps a jagged `row_counts` layout across a fold, and the
even-prepad/folded-count recurrence. Both are host-side (numpy / pure Python) so
the schedule is precomputed and baked into the frx trace as a constant. Shared
by the jagged round schedule and the LogUp-GKR circuit transition.
"""
from __future__ import annotations

import numpy as np


def _segment_gather_np(
    src_counts: tuple[int, ...], dst_counts: tuple[int, ...]
) -> np.ndarray | None:
    """Numpy core of the segment gather.

    Gather indices remapping a jagged layout from `src_counts` to `dst_counts`;
    positions past a segment's source rows get the sentinel `sum(src_counts)`
    (a downstream `_gather_pad` resolves it to the padding value). None when the
    layouts already agree. Stays numpy so the schedule is precomputed host-side
    and baked into the `frx.jit` trace as a constant, where an `np.asarray` of a
    fnp value would trip on a tracer.
    """
    if src_counts == dst_counts:
        return None
    sentinel = sum(src_counts)
    gather = np.full(sum(dst_counts), sentinel, dtype=np.int32)
    src_pos = dst_pos = 0
    # strict: a silently truncated zip would emit a sentinel-filled (all
    # padding) gather instead of failing.
    for src, dst in zip(src_counts, dst_counts, strict=True):
        copy = min(src, dst)
        gather[dst_pos : dst_pos + copy] = np.arange(src_pos, src_pos + copy)
        src_pos += src
        dst_pos += dst
    return gather


def _prepad_folded(
    row_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One transition's even-prepad counts and the resulting folded counts.

    Odd segments pad up to even so the stride-2 fold never pairs across a
    batch boundary; the fold then halves the padded count. One recurrence
    so the wrapper's truncation guard validates exactly the schedule the core's
    gathers fold to.
    """
    prepad = tuple(rc + rc % 2 for rc in row_counts)
    folded = tuple(pc // 2 for pc in prepad)
    return prepad, folded
