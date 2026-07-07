# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic dataclasses of the jagged sumcheck round engine: the
fixed-width round caps, the interpolation constants, and the static round
schedule (the LogUp-specific state/planes live in
`zorch.logup_gkr._jagged_types`)."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class RoundWidthCaps:
    """Fixed round-buffer widths for the size-invariant jagged sumcheck
    (xla#179): with caps set, every round of a phase runs at one static
    operand shape -- the live prefix tracked by the round's `live` operand --
    so one compiled round kernel serves every round, layer, and shard under
    the caps. Hashable (a jit static arg on the per-layer round zone).

    `row` bounds the row-phase plane/gather width (>= the round-0 even-padded
    layout, a multiple of 4); `eq_row` bounds the row-eq table (>= 2^nrv,
    even); `interaction` bounds the dense-phase state and eq width (>= 2^niv,
    a multiple of 4)."""

    row: int
    eq_row: int
    interaction: int


@dataclass(frozen=True)
class _InterpConsts:
    """The Lagrange interpolation constants (the `{0..DEGREE}` natural domain and
    the inverse Vandermonde). They depend only on dtype, so the round kernels bake
    them in as closure constants -- NOT export operands, so not a pytree."""

    naturals: Array
    inv_vand: Array


@dataclass(frozen=True)
class _JaggedSchedule:
    """A jagged layer's static round schedule: the layer's `row_counts`
    operand plus the per-round live triples (the re-pad schedule itself
    derives in-kernel from these), the interpolation constants, and
    the batch/row variable counts plus the challenge limb count. Rides beside
    the state so the loop signatures stay `(state, schedule, transcript)`
    rather than a positional-arg bag. `caps` selects the fixed-width round
    layout; None runs the exact (per-round-shape) layout, whose static padded
    widths ride `out_pairs` (None under caps — width-preserving). `meta` (the
    host-built explicit schedule) feeds ONLY the reference oracle
    `_run_jagged_rounds_reference`; the round loop never reads it."""

    row_counts: Array
    live: list[Array]
    out_pairs: tuple[int, ...] | None
    consts: _InterpConsts
    nrv: int
    niv: int
    challenge_limbs: int
    caps: RoundWidthCaps | None = None
    meta: list[tuple[Array | None, Array, Array, Array]] | None = None
