# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR fractional-sum circuit.

A layer is four equal-length MLEs `(n0, n1, d0, d1)`: index `i` carries two
binary-tree children, the fractions `n0[i]/d0[i]` and `n1[i]/d1[i]`. A
transition folds each pair into one fraction --

    n0/d0 + n1/d1 = (n0*d1 + n1*d0) / (d0*d1)

-- pairing LSB-adjacent nodes (stride 2), which halves every MLE and eliminates
one (row) variable. Iterating to the interaction floor (`num_row_variables == 0`)
leaves one fraction per interaction; `extract_outputs` interleaves the two
children back into the output numerator/denominator MLEs over the interaction
variables plus one.

Two layouts share that fold. Dense (`GkrLayer`): every interaction has one row
count, so a layer is a flat power of two with no padding. Jagged
(`JaggedGkrLayer`): the MLEs are stored interaction-major with per-interaction
row counts; a transition pre-pads odd segments and post-pads to a
consumer-supplied schedule with the additive-identity fraction (n=0, d=1), so
the flat stride-2 fold never pairs across an interaction boundary. Which
heights the schedule pads to is the consumer's policy, and interaction
fingerprinting likewise stays in the consumer -- zorch stays scheme-agnostic.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class GkrLayer:
    """One dense fractional-sum layer over (interaction || row) variables.

    `num_interaction_variables` is the floor: folding stops once the row
    variables are exhausted and only the interaction dimension remains.
    """

    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array
    num_interaction_variables: int

    def __post_init__(self) -> None:
        # Reject malformed layers at construction rather than at a later
        # broadcast or negative-row-count. num_variables also checks the
        # power-of-two width via log2_strict_usize.
        shape = self.numerator_0.shape
        for name in ("numerator_1", "denominator_0", "denominator_1"):
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"all MLEs must share a shape; {name} is "
                    f"{getattr(self, name).shape}, numerator_0 is {shape}"
                )
        if not 0 <= self.num_interaction_variables <= self.num_variables:
            raise ValueError(
                f"num_interaction_variables must be in [0, {self.num_variables}], "
                f"got {self.num_interaction_variables}"
            )

    @property
    def num_variables(self) -> int:
        return log2_strict_usize(self.numerator_0.shape[0])

    @property
    def num_row_variables(self) -> int:
        return self.num_variables - self.num_interaction_variables


@dataclass(frozen=True)
class LogUpGkrOutput:
    """Final numerator/denominator MLEs after all transitions."""

    numerator: Array
    denominator: Array


def _fold_pairs(
    n0: Array, n1: Array, d0: Array, d1: Array
) -> tuple[Array, Array, Array, Array]:
    """Stride-2 fraction-pair fold shared by the dense and jagged transitions.

    Even nodes become the next layer's 0-child, odd nodes its 1-child. Halves
    every MLE.
    """
    n0e, n0o = n0[0::2], n0[1::2]
    n1e, n1o = n1[0::2], n1[1::2]
    d0e, d0o = d0[0::2], d0[1::2]
    d1e, d1o = d1[0::2], d1[1::2]
    return (
        n0e * d1e + n1e * d0e,
        n0o * d1o + n1o * d0o,
        d0e * d1e,
        d0o * d1o,
    )


def layer_transition(layer: GkrLayer) -> GkrLayer:
    """Fold one row variable: sum each LSB-adjacent fraction pair (stride 2).

    Requires a row variable to fold.
    """
    if layer.num_row_variables < 1:
        raise ValueError("no row variable left to fold")
    rn0, rn1, rd0, rd1 = _fold_pairs(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )
    return GkrLayer(
        numerator_0=rn0,
        numerator_1=rn1,
        denominator_0=rd0,
        denominator_1=rd1,
        num_interaction_variables=layer.num_interaction_variables,
    )


def build_pyramid(first: GkrLayer) -> list[GkrLayer]:
    """Eager fold from `first` (most row variables) down to the interaction floor.

    Eager rather than one fused program: the full pyramid does not fit one
    `@jit` at scale, and each transition's output feeds the next layer's
    per-variable sumcheck independently.
    """
    layers = [first]
    while layers[-1].num_row_variables > 0:
        layers.append(layer_transition(layers[-1]))
    return layers


def _interleave(child_0: Array, child_1: Array) -> Array:
    """Interleave two equal-length children into [c0[0], c1[0], c0[1], c1[1], ...]."""
    return jnp.stack([child_0, child_1], axis=-1).flatten()


def extract_outputs(layer: GkrLayer) -> LogUpGkrOutput:
    """Interleave the two children of the floor layer into the output MLEs.

    At `num_row_variables == 0` each index is one interaction's fraction with a
    0-child and a 1-child; interleaving recovers the MLE over the interaction
    variables plus one.
    """
    if layer.num_row_variables != 0:
        raise ValueError(
            f"extract_outputs expects the interaction floor (num_row_variables "
            f"== 0), got {layer.num_row_variables}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )


@dataclass(frozen=True)
class JaggedGkrLayer:
    """One jagged fractional-sum layer, stored interaction-major.

    The four MLEs are flat over `sum(row_counts)`: all rows of interaction 0,
    then interaction 1, and so on, so each interaction carries its own row
    count. The interaction count must be a power of two (the consumer pads its
    interaction list). Row counts are static Python ints, so the transition's
    gather indices are built at trace time.
    """

    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array
    row_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(rc < 1 for rc in self.row_counts):
            raise ValueError(f"row counts must be >= 1, got {self.row_counts}")
        log2_strict_usize(self.num_interactions)
        height = self.height
        for name in (
            "numerator_0",
            "numerator_1",
            "denominator_0",
            "denominator_1",
        ):
            shape = getattr(self, name).shape
            if shape != (height,):
                raise ValueError(
                    f"each MLE must be flat over sum(row_counts) == {height}; "
                    f"{name} is {shape}"
                )

    @property
    def num_interactions(self) -> int:
        return len(self.row_counts)

    @property
    def num_interaction_variables(self) -> int:
        return log2_strict_usize(self.num_interactions)

    @property
    def height(self) -> int:
        return sum(self.row_counts)

    @property
    def start_indices(self) -> tuple[int, ...]:
        """Prefix sums of row_counts: segment i spans [start[i], start[i+1])."""
        return (0, *itertools.accumulate(self.row_counts))


def _segment_gather(
    src_counts: tuple[int, ...], dst_counts: tuple[int, ...]
) -> Array | None:
    """Gather indices remapping a jagged layout from src_counts to dst_counts.

    Positions past a segment's source rows get the sentinel `sum(src_counts)`,
    which `_gather_pad` resolves to the padding value. None when the layouts
    already agree (no gather needed).
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
    return jnp.asarray(gather)


def _gather_pad(arr: Array, gather: Array, pad_val: int) -> Array:
    """Gather with sentinel-aware padding.

    Clamped take + where rather than appending the padding value to `arr`,
    which would copy the whole array.
    """
    is_pad = gather >= arr.shape[0]
    clamped = jnp.minimum(gather, arr.shape[0] - 1)
    return jnp.where(is_pad, pad_val, arr[clamped])


def _pad_neutral(
    n0: Array, n1: Array, d0: Array, d1: Array, gather: Array | None
) -> tuple[Array, Array, Array, Array]:
    """Re-pad the four MLEs to `gather`'s layout with the additive-identity
    fraction (n=0, d=1). No-op when the layouts already agree (`gather` None).
    """
    if gather is None:
        return n0, n1, d0, d1
    return (
        _gather_pad(n0, gather, 0),
        _gather_pad(n1, gather, 0),
        _gather_pad(d0, gather, 1),
        _gather_pad(d1, gather, 1),
    )


def _pad_to_width(arr: Array, width: int, neutral: int) -> Array:
    """Extend `arr` to `width` with the fold-neutral fraction tail -- 0 for a
    numerator, 1 for a denominator -- keeping the live prefix at the front. The
    fixed-width-buffer convention the rolled jagged scans carry so a chain of
    differently-sized layers stays shape-invariant across the scan."""
    pad = width - arr.shape[0]
    if pad == 0:
        return arr
    tail = jnp.zeros((pad,), arr.dtype) if neutral == 0 else jnp.ones((pad,), arr.dtype)
    return jnp.concatenate([arr, tail])


def _fixed_width_gather(
    src_counts: tuple[int, ...], dst_counts: tuple[int, ...], width: int
) -> np.ndarray:
    """`_segment_gather` laid into a fixed `width` buffer -- its companion for the
    rolled scans, which carry every layer in one width. `_segment_gather`'s
    intra-segment sentinel is `sum(src_counts)`: past the live rows in the
    exactly-sized layout, but a live slot in the wider buffer, so remap it (and
    any index past the live rows) to `width`, which `_gather_pad` resolves to the
    neutral pad rather than a stale slot. `None` (layouts already agree) becomes
    the identity over the live prefix."""
    live = sum(src_counts)
    seg = _segment_gather(src_counts, dst_counts)
    base = np.arange(live, dtype=np.int32) if seg is None else np.asarray(seg, np.int32)
    base = np.where(base >= live, width, base)
    row = np.full(width, width, dtype=np.int32)
    row[: base.shape[0]] = base
    return row


def jagged_layer_transition(
    layer: JaggedGkrLayer, out_row_counts: tuple[int, ...]
) -> JaggedGkrLayer:
    """Fold one row variable per segment, then pad to the consumer's schedule.

    Odd segments are pre-padded with the additive-identity fraction (n=0, d=1)
    so the flat stride-2 fold never pairs across an interaction boundary; the
    folded segments are then padded out to `out_row_counts` with the same
    neutral rows. The schedule is the consumer's policy; this block only
    refuses to truncate, which would silently drop fractions from the sum.
    """
    if len(out_row_counts) != layer.num_interactions:
        raise ValueError(
            f"schedule must cover all {layer.num_interactions} interactions, "
            f"got {len(out_row_counts)} entries"
        )
    # Odd segments pad up to even so the stride-2 fold can't pair across an
    # interaction boundary; the folded count is then half the padded one.
    prepad_counts = tuple(rc + rc % 2 for rc in layer.row_counts)
    folded_counts = tuple(pc // 2 for pc in prepad_counts)
    for i, (folded, out) in enumerate(zip(folded_counts, out_row_counts)):
        if out < folded:
            raise ValueError(
                f"schedule truncates interaction {i}: folded row count "
                f"{folded} > target {out}"
            )

    n0, n1, d0, d1 = _pad_neutral(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        _segment_gather(layer.row_counts, prepad_counts),
    )
    rn0, rn1, rd0, rd1 = _fold_pairs(n0, n1, d0, d1)
    rn0, rn1, rd0, rd1 = _pad_neutral(
        rn0, rn1, rd0, rd1, _segment_gather(folded_counts, out_row_counts)
    )

    return JaggedGkrLayer(
        numerator_0=rn0,
        numerator_1=rn1,
        denominator_0=rd0,
        denominator_1=rd1,
        row_counts=out_row_counts,
    )


def extract_jagged_outputs(layer: JaggedGkrLayer) -> LogUpGkrOutput:
    """Interleave the floor layer's children into the output MLEs.

    The floor is row counts all 1 -- one fraction pair per interaction -- the
    jagged dual of `extract_outputs`'s `num_row_variables == 0` precondition.
    A schedule that stops higher folds the rest down with
    `jagged_layer_transition` first; how far to fold is the consumer's call.
    """
    if any(rc != 1 for rc in layer.row_counts):
        raise ValueError(
            f"extract_jagged_outputs expects the interaction floor (row "
            f"counts all 1), got {layer.row_counts}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )


def scan_build_jagged_pyramid(
    first: JaggedGkrLayer, schedules: Sequence[tuple[int, ...]]
) -> list[JaggedGkrLayer]:
    """Build the jagged pyramid via one `lax.scan` over the transitions -- fused
    (one traced region, O(1) in the pyramid depth), not the eager Python loop of
    `jagged_layer_transition` dispatches that scales the runtime with the layer
    count. `schedules[k]` is transition `k`'s `out_row_counts` (the consumer's
    halving policy, the same argument the eager transition takes); the planes ride
    a fixed-width buffer with the neutral fraction padding the dead tail so the
    scan carry stays shape-invariant. Byte-identical to iterating
    `jagged_layer_transition` down the chain; returns `[first, ..., floor]`
    (sp1-zorch#55)."""
    schedules = list(schedules)
    # A chain that is already at the floor has no transition to fold; `lax.scan`
    # with length 0 is illegal, so short-circuit before building any xs.
    if not schedules:
        return [first]

    # Walk the chain host-side to recover every transition's static layout:
    # the row counts entering transition k, the even prepad counts the fold
    # needs, and the postpad target `schedules[k]`. The gathers are pure
    # functions of these Python ints, so the whole xs stack is built at trace
    # time.
    src_counts: list[tuple[int, ...]] = [first.row_counts, *schedules]
    prepad_counts = [tuple(rc + rc % 2 for rc in c) for c in src_counts[:-1]]
    folded_counts = [tuple(pc // 2 for pc in counts) for counts in prepad_counts]

    # The planes ride a fixed-width buffer (live prefix front, neutral tail) so
    # the scan carry stays shape-invariant. A transition's prepad buffer is the
    # widest intermediate -- an odd segment padding up to even makes it at least
    # as wide as its own (and thus every smaller) layer height -- so the max
    # prepad height alone sizes the buffer.
    plane_width = max(sum(counts) for counts in prepad_counts)
    # Per-transition prepad / postpad gathers, each `_segment_gather` laid into
    # the fixed width and stacked as the scan xs.
    prepad_gathers = jnp.asarray(
        np.stack(
            [
                _fixed_width_gather(src_counts[k], prepad_counts[k], plane_width)
                for k in range(len(schedules))
            ]
        )
    )
    postpad_gathers = jnp.asarray(
        np.stack(
            [
                _fixed_width_gather(folded_counts[k], schedules[k], plane_width)
                for k in range(len(schedules))
            ]
        )
    )

    init = (
        _pad_to_width(first.numerator_0, plane_width, 0),
        _pad_to_width(first.numerator_1, plane_width, 0),
        _pad_to_width(first.denominator_0, plane_width, 1),
        _pad_to_width(first.denominator_1, plane_width, 1),
    )

    def step(
        carry: tuple[Array, Array, Array, Array],
        gathers: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
        n0, n1, d0, d1 = carry
        prepad, postpad = gathers
        # One eager transition inside the fixed width: prepad odd segments to
        # even, fold stride-2 over the (now even) live prefix, then postpad the
        # folded segments to the schedule. The dead tail is the neutral fraction
        # throughout -- it folds to neutral and the postpad gather discards it --
        # so the live region matches the eager exact-sized transition.
        n0, n1, d0, d1 = _pad_neutral(n0, n1, d0, d1, prepad)
        n0, n1, d0, d1 = _fold_pairs(n0, n1, d0, d1)
        # The fold halves the buffer, so re-pad the four planes back up to the
        # fixed width with the neutral tail before the postpad gather (indexed
        # against the full width) and before the next carry.
        n0, n1, d0, d1 = (
            _pad_to_width(n0, plane_width, 0),
            _pad_to_width(n1, plane_width, 0),
            _pad_to_width(d0, plane_width, 1),
            _pad_to_width(d1, plane_width, 1),
        )
        n0, n1, d0, d1 = _pad_neutral(n0, n1, d0, d1, postpad)
        # A dead-fold + re-pad chain can alias into another scan iteration under
        # XLA's scan-body value numbering; barrier the per-step carry so each
        # transition's planes stay independent (the jagged_prover mitigation).
        n0, n1, d0, d1 = lax.optimization_barrier((n0, n1, d0, d1))
        out = (n0, n1, d0, d1)
        return out, out

    _, stacked = lax.scan(step, init, (prepad_gathers, postpad_gathers))

    layers = [first]
    for k, out_row_counts in enumerate(schedules):
        height = sum(out_row_counts)
        layers.append(
            JaggedGkrLayer(
                numerator_0=stacked[0][k][:height],
                numerator_1=stacked[1][k][:height],
                denominator_0=stacked[2][k][:height],
                denominator_1=stacked[3][k][:height],
                row_counts=out_row_counts,
            )
        )
    return layers
