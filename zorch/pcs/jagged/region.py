# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged region packing for the SP1 trace commit.

A region is the committable form of one shard's variable-height chip traces:
every chip column-major-flattened into one dense buffer, end-padded with zeros
to a multiple of the stacking height. ``row_counts``/``column_counts`` follow
SP1's structure-hash convention — per-chip entries first, then a trailing
``(max_height, leftover)`` / ``(num_added_cols - 1, 1)`` pair that decodes the
trailing zero pad — and are bound into the commitment hash, so their encoding is
part of the commitment format (it lives with this PCS's commit, not the generic
``zorch/commit`` blocks), not a packing detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

import frx
import frx.numpy as jnp
from frx import Array


def structure_counts(
    heights: Sequence[int],
    widths: Sequence[int],
    *,
    log_stacking_height: int,
    max_log_row_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    """SP1's structure-hash ``(row_counts, column_counts)`` for a region of the
    given chip heights/widths, plus the raw packed area and its
    stacking-aligned size: per-chip entries first, then the trailing pad pair
    decoding the end pad to the stacking alignment. One definition shared by
    the prover's region packing and the jagged-eval verifier dual's column
    manifest — the counts and the alignment are part of the commitment
    format, so the two sides must derive them identically."""
    S = 1 << log_stacking_height
    max_height = 1 << max_log_row_count
    total_area = sum(int(h) * int(w) for h, w in zip(heights, widths, strict=True))
    # End-pad to the next multiple of S (at least one full stack); the pad
    # decodes as full max_height columns plus one leftover column.
    aligned = max(((total_area + S - 1) // S) * S, S)
    num_added_vals = aligned - total_area
    num_added_cols = max((num_added_vals + max_height - 1) // max_height, 1)
    leftover = num_added_vals - (num_added_cols - 1) * max_height
    row_counts = tuple(int(h) for h in heights) + (max_height, int(leftover))
    column_counts = tuple(int(w) for w in widths) + (int(num_added_cols - 1), 1)
    return row_counts, column_counts, total_area, aligned


@partial(frx.jit, static_argnames=("num_added_vals", "pad_dtype"))
def _pack_chip_data(
    chips: tuple[Array, ...], *, num_added_vals: int, pad_dtype: Any
) -> Array:
    """Column-major flatten each chip and concat with the trailing zero pad.

    Under @jit XLA writes every chip directly into its slot of the output
    buffer; eagerly each ``chip.T.reshape(-1)`` is its own dispatch plus a
    full extra copy at the concat.
    """
    flats = [chip.T.reshape(-1) for chip in chips]
    if num_added_vals > 0:
        flats.append(jnp.zeros(num_added_vals, dtype=pad_dtype))
    return jnp.concatenate(flats)


# Pytree: `dense` is the only array leaf; the SMCS row/column counts, stacking
# height, and names are static aux. Lets a region cross the chain's @jit
# boundary as part of a donatable carry.
@partial(
    frx.tree_util.register_dataclass,
    data_fields=["dense"],
    meta_fields=[
        "chip_starts",
        "row_counts",
        "column_counts",
        "log_stacking_height",
        "chip_names",
    ],
)
@dataclass(frozen=True)
class JaggedRegion:
    """One committable region: dense buffer + SMCS row/column counts.

    ``chip_starts`` (cumulative chip-area offsets, length ``num_chips + 1``)
    addresses each chip's raw, unpadded data inside ``dense``; zero-height
    chips occupy no dense data but keep their counts entry so chip indexing
    matches the shard's chip order.
    """

    dense: Array
    chip_starts: tuple[int, ...]
    row_counts: tuple[int, ...]
    column_counts: tuple[int, ...]
    log_stacking_height: int
    chip_names: tuple[str, ...] = ()

    @property
    def num_chips(self) -> int:
        return len(self.row_counts) - 2

    @property
    def chip_heights(self) -> tuple[int, ...]:
        return self.row_counts[:-2]

    @property
    def chip_widths(self) -> tuple[int, ...]:
        return self.column_counts[:-2]

    @property
    def raw_size(self) -> int:
        """Cumulative chip data size, excluding the trailing zero pad."""
        return int(self.chip_starts[-1])

    @property
    def block(self) -> Array:
        """The ``[K, S]`` message-domain matrix (row ``k`` is stacked dense
        block ``k``) — a free reshape view of ``dense``. The layout the commit
        encodes and the stacked open consumes (``StackedRound.block``)."""
        S = 1 << self.log_stacking_height
        if self.dense.shape[0] % S != 0:
            raise ValueError(
                f"dense size {self.dense.shape[0]} must be a multiple of the "
                f"stacking height {S} (from_chips pads to it)"
            )
        return self.dense.reshape(-1, S)

    @classmethod
    def from_chips(
        cls,
        chips: Sequence[Array],
        *,
        log_stacking_height: int,
        max_log_row_count: int,
        chip_names: Optional[Sequence[str]] = None,
    ) -> "JaggedRegion":
        if not chips:
            raise ValueError("JaggedRegion.from_chips: empty chip list")
        if chip_names is not None and len(chip_names) != len(chips):
            raise ValueError(
                f"JaggedRegion.from_chips: chip_names length {len(chip_names)} "
                f"!= chips length {len(chips)}"
            )
        max_height = 1 << max_log_row_count

        heights: list[int] = []
        widths: list[int] = []
        starts: list[int] = [0]
        nonempty: list[Array] = []
        total_area = 0
        # One dtype for the whole region: a mixed pair would silently promote
        # through the concat and change the commitment preimage.
        dtype = chips[0].dtype
        for chip in chips:
            if chip.ndim != 2:
                raise ValueError(
                    f"chips must be 2-D (rows, cols) arrays, got shape {chip.shape}"
                )
            if chip.dtype != dtype:
                raise ValueError(
                    f"all chips must share dtype {dtype}, got {chip.dtype}"
                )
            h, w = chip.shape
            if h > max_height:
                raise ValueError(
                    f"chip row_count {h} exceeds max {max_height} "
                    f"(max_log_row_count={max_log_row_count})"
                )
            heights.append(int(h))
            widths.append(int(w))
            if h > 0 and w > 0:
                nonempty.append(chip)
                total_area += int(h) * int(w)
            starts.append(total_area)

        row_counts, column_counts, _, aligned = structure_counts(
            heights,
            widths,
            log_stacking_height=log_stacking_height,
            max_log_row_count=max_log_row_count,
        )
        dense = _pack_chip_data(
            tuple(nonempty), num_added_vals=int(aligned - total_area), pad_dtype=dtype
        )

        return cls(
            dense=dense,
            chip_starts=tuple(starts),
            row_counts=row_counts,
            column_counts=column_counts,
            log_stacking_height=log_stacking_height,
            chip_names=tuple(chip_names) if chip_names is not None else (),
        )
