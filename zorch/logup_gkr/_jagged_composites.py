# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`zorch.sumcheck.round` marker composites: each wraps one
eager round body so a recognizing emitter fuses it while an unclaimed
marker decomposes inline, byte-identical."""

from __future__ import annotations

from functools import partial

import frx.numpy as fnp
from frx import Array
from hash_frx._composite import composite

from zorch.logup_gkr._jagged_rounds import (
    _bind_planes,
    _fix_last,
    _round_poly_int,
    _round_poly_row,
)
from zorch.logup_gkr._jagged_types import _DEGREE, _Planes, _RoundScalars
from zorch.sumcheck.domain import fold
from zorch.sumcheck.jagged.buffers import _pad_to_width
from zorch.sumcheck.jagged.schedule import _derive_row_schedule
from zorch.sumcheck.jagged.types import _InterpConsts
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    SUMCHECK_ROUND_MARKER_VERSION,
)

# --- FS-less compute-only round composites -----------------------------------
# The host loop wraps each round's fold+sum in a `zorch.sumcheck.round` marker;
# Fiat-Shamir stays the separate `hash_frx.poseidon2` composite the transcript emits
# between rounds. When no emitter claims the marker (CPU, or a pre-#327 pin), the
# `lax.composite` decomposition runs inline, so the marked path is byte-identical
# to the eager body. All five round bodies are marked: first (round 0, jagged),
# mid row (jagged), boundary (the row->interaction handoff), mid interaction
# (dense), and final (the `_fix_last` fold) -- selected by `_run_jagged_rounds`
# / `_finalize_layer` per round.


def _round_composite_dense_decomp(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    **_attrs: object,
) -> tuple[Array, _Planes, Array]:
    """The `zorch.sumcheck.round` decomposition for the `dense` (interaction) `mid`
    phase -- the byte-exact fallback a recognizing emitter replaces. `_attrs`
    (phase / variant / degree / poly_form) are composite metadata the emitter
    parses; the decomposition needs only the operands. The interp constants ride
    as operands (`naturals` / `inv_vand`) rather than a lifted closure -- the
    emitter may instead rebuild them from `degree` + `poly_form` and drop the two
    trailing operands.

    Width-preserving (size-invariant): the state enters live to
    `4 * live[0]` elements (`live[0]` = the round's live reduce pairs) in
    width-`m` buffers and leaves live to `2 * live[0]` in the SAME width -- the
    fold's halved output zero-pads back to `m`, and the sum masks to the live
    pairs. One operand/result shape therefore serves every round of the phase;
    on the exact layout (`4 * live[0] == m`) the live prefix is the whole
    buffer and the values match `_fix_and_sum_int` bit-for-bit."""
    width = planes.n0.shape[0]
    bound = _bind_planes(planes, alpha)
    eq_bound = fold(eq_int, alpha, msb=False)
    poly = _round_poly_int(
        bound,
        eq_bound,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    out = _Planes(
        *(_pad_to_width(a, width, 0) for a in (bound.n0, bound.n1, bound.d0, bound.d1))
    )
    return poly, out, _pad_to_width(eq_bound, width, 0)


def _composite_fix_and_sum_dense(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=mid, variant=dense) marker
    around the uniform interaction fold+sum -- a batched LogUp-GKR round (the
    hardcoded LogUp combine + the eq_adj/pad_adj virtual-mass correction, not a
    plain product). The signature mirrors `_fix_and_sum_int` (interp consts
    threaded as operands, plus the trailing i32[2] `live` prefix marker) so the
    round loop can select it in place. No `challenge_limbs`: the fold challenge
    `alpha` arrives pre-recomposed as one operand whose dtype carries base vs
    extension. Width-preserving: the folded state returns at the input width,
    live to `2 * live[0]`. Byte-identical to `_fix_and_sum_int` on the live
    prefix whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    return composite(
        _round_composite_dense_decomp,
        planes,
        eq_int,
        alpha,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_row_decomp(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    out_pairs: int | None = None,
    **_attrs: object,
) -> tuple[Array, _Planes, Array]:
    """The `zorch.sumcheck.round` decomposition for the `jagged` (row) `mid` phase
    -- the byte-exact fallback a recognizing emitter replaces. `_attrs` (phase /
    variant / degree / poly_form) are composite metadata the emitter parses; the
    decomposition needs only the operands. The device-derived schedule: the re-pad
    schedule derives in-trace from `row_counts` + the
    round index `live[2]` (`_derive_row_schedule` — the claimed kernel runs
    the same derivation in place), so no index array is uploaded per round.
    `out_pairs` is the exact layout's STATIC padded pair count, closed over by
    the wrapper (not an operand); None selects the width-preserving capped
    convention (out width = the plane buffer width).

    Size-invariance: the sum masks to the `live[0]` live pairs (the
    re-padded state's live prefix is `2 * live[0]`, the rest of the derived
    gather is sentinel → neutral pad), and the folded `eq_row` zero-pads back
    to its input width, live to `live[1] // 2`. On the exact layout the live
    prefixes are the whole buffers and the values match `_fix_and_sum_row`
    bit-for-bit (modulo the eq_row width restore)."""
    eq_width = eq_row.shape[0]
    folded_len = planes.n0.shape[0] // 2
    num_pairs = folded_len if out_pairs is None else out_pairs
    gather, col_index, pair_index = _derive_row_schedule(
        row_counts, live[2], num_pairs, sentinel=folded_len, idx_dtype=fnp.int32
    )
    bound = _bind_planes(planes, alpha)
    eq_bound = fold(eq_row, alpha, msb=False)
    poly, padded = _round_poly_row(
        bound,
        gather,
        col_index,
        pair_index,
        eq_bound,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    return poly, padded, _pad_to_width(eq_bound, eq_width, 0)


def _composite_fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=mid, variant=jagged) marker
    around the row-variable fold+sum -- the segment-based jagged round (the
    hardcoded LogUp combine over the derived re-pad schedule and the
    segment-local `eq_row`, not a plain product). The signature mirrors
    `_fix_and_sum_row` with the schedule operands replaced by the layer's
    `row_counts` (i32[nseg], device-derived schedule) plus the trailing i32[3]
    `{live pairs, live eq_row, round}` operand. `out_pairs` (static, exact
    layout only) sizes the padded output run; None keeps the capped
    width-preserving convention. `eq_row` returns width-preserved (folded
    live prefix, zero tail). Byte-identical to `_fix_and_sum_row` on the live
    prefixes whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    decomp = partial(_round_composite_row_decomp, out_pairs=out_pairs)
    return composite(
        decomp,
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="jagged",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_final_decomp(
    planes: _Planes,
    alpha: Array,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array]:
    """The `zorch.sumcheck.round` decomposition for the `final` phase -- the
    byte-exact fallback a recognizing emitter replaces. Fold only: bind the
    final challenge and read off the four pair openings. No round poly, no eq,
    no interp constants -- nothing sums, so the marker carries just the
    length-2 planes and `alpha`."""
    return _fix_last(planes, alpha)


def _composite_fix_last(
    planes: _Planes, alpha: Array
) -> tuple[Array, Array, Array, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=final, variant=dense)
    marker around the layer tail's final fold (`_fix_last`): bind the last
    challenge and emit the four scalar pair openings. The signature mirrors
    `_fix_last` so `_finalize_layer` can select it in place. Byte-identical to
    `_fix_last` whenever the marker is unclaimed (`lax.composite` runs the
    decomposition)."""
    return composite(
        _round_composite_final_decomp,
        planes,
        alpha,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="final",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )


def _round_composite_boundary_decomp(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    **_attrs: object,
) -> tuple[Array, _Planes]:
    """The `zorch.sumcheck.round` decomposition for the `boundary` (row ->
    interaction handoff) phase -- the byte-exact fallback a recognizing emitter
    replaces. The operand layout is the dense `mid` ABI, but `eq_int` enters at
    2x the post-bind state width and is NOT folded: only the planes bind by
    `alpha`. `eq_int` is dropped from the outputs -- it rides through the round
    unchanged, so emitting it would copy a full tensor per boundary round; the
    wrapper returns the caller's own `eq_int` instead.

    Unlike the width-preserving dense `mid`, the boundary keeps its halving
    output (planes leave at half the input width, live to `2 * live[0]`): it
    fires once per layer, and the halved width is what keeps the plane/eq
    widths equal through the dense phase on the exact layout. The fixed-width
    route slices the output down to its interaction cap outside the marker."""
    bound = _bind_planes(planes, alpha)
    poly = _round_poly_int(
        bound,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )
    return poly, bound


def _composite_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=boundary, variant=dense)
    marker around the row->interaction handoff -- bind the last row variable's
    challenge (row-shaped fold), then the first interaction round's univariate
    over the still-unfolded `eq_int`. The signature mirrors
    `_fix_and_sum_boundary` (plus the trailing i32[2] `live` prefix marker) so
    the round loop can select it in place; the unchanged `eq_int` is returned
    from outside the marker (not a composite output). Byte-identical to
    `_fix_and_sum_boundary` on the live prefix whenever the marker is unclaimed
    (`lax.composite` runs the decomposition)."""
    poly, planes = composite(
        _round_composite_boundary_decomp,
        planes,
        eq_int,
        alpha,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="boundary",
        variant="dense",
        degree=_DEGREE,
        poly_form="coefficient",
    )
    return poly, planes, eq_int


def _round_composite_first_row_decomp(
    planes: _Planes,
    eq_row: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    out_pairs: int | None = None,
    **_attrs: object,
) -> tuple[Array, _Planes]:
    """The `zorch.sumcheck.round` decomposition for the `jagged` (row) `first`
    phase -- the byte-exact fallback a recognizing emitter replaces. The operand
    order is the `mid` row ABI minus `alpha`: round 0 binds nothing, so there is
    no previous challenge to fold by and the marker carries no `alpha` slot.
    The re-pad schedule derives in-trace from `row_counts` + `live[2]` (= 0,
    see the row `mid` decomposition); the derived indexes span the
    FULL raw height (nothing folds this round). The sum masks to the `live[0]`
    live pairs of a fixed-width schedule (the re-padded state's live prefix is
    `2 * live[0]`)."""
    raw_len = planes.n0.shape[0]
    num_pairs = raw_len // 2 if out_pairs is None else out_pairs
    gather, col_index, pair_index = _derive_row_schedule(
        row_counts, live[2], num_pairs, sentinel=raw_len, idx_dtype=fnp.int32
    )
    return _round_poly_row(
        planes,
        gather,
        col_index,
        pair_index,
        eq_row,
        eq_int,
        scalars,
        _InterpConsts(naturals, inv_vand),
        live_pairs=live[0],
    )


def _composite_sum_as_poly_row(
    planes: _Planes,
    row_counts: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes]:
    """Emit the FS-less `zorch.sumcheck.round` (phase=first, variant=jagged)
    marker around the round-0 sum -- no fold, no challenge, just the row-shaped
    round poly over the raw layer plus the re-padded state the caller binds next
    round. The signature mirrors `_round_poly_row` with the schedule operands
    replaced by the layer's `row_counts` plus the trailing i32[3]
    `{live pairs, live eq_row, round}` operand, so the round loop can select it
    in the `sum0` slot. `out_pairs` (static, exact layout only) sizes the
    padded output run; None keeps the capped width-preserving convention.
    Byte-identical to `_round_poly_row` on the live prefixes whenever the
    marker is unclaimed (`lax.composite` runs the decomposition)."""
    decomp = partial(_round_composite_first_row_decomp, out_pairs=out_pairs)
    return composite(
        decomp,
        planes,
        eq_row,
        row_counts,
        eq_int,
        scalars,
        consts.naturals,
        consts.inv_vand,
        live,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="first",
        variant="jagged",
        degree=_DEGREE,
        poly_form="coefficient",
    )
