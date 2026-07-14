# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR fractional-sum circuit.

A layer is four equal-length MLEs `(n0, n1, d0, d1)`: index `i` carries two
binary-tree children, the fractions `n0[i]/d0[i]` and `n1[i]/d1[i]`. A
transition folds each pair into one fraction --

    n0/d0 + n1/d1 = (n0*d1 + n1*d0) / (d0*d1)

-- pairing LSB-adjacent nodes (stride 2), which halves every MLE and eliminates
one (row) variable. Iterating to the batch floor (`num_row_variables == 0`)
leaves one fraction per batch element; `extract_outputs` interleaves the two
children back into the output numerator/denominator MLEs over the batch
variables plus one.

The four MLEs share a length, but the numerator pair and the denominator pair
need not share a field: a base-field numerator under an extension-field
denominator promotes to their common field at the fold (`n0*d1 + n1*d0`), so a
consumer can keep a layer's numerator reads narrow until its first transition.
zorch stays scheme-agnostic about why a consumer would; it only guarantees the
promotion is byte-identical to folding an all-extension copy.

Two layouts share that fold. Dense (`GkrLayer`): every batch element has one row
count, so a layer is a flat power of two with no padding. Jagged
(`JaggedGkrLayer`): the MLEs are stored batch-major with per-batch-element
row counts; a transition pre-pads odd segments and post-pads to a
consumer-supplied schedule with the additive-identity fraction (n=0, d=1), so
the flat stride-2 fold never pairs across a batch boundary. Which
heights the schedule pads to is the consumer's policy, and interaction
fingerprinting likewise stays in the consumer -- zorch stays scheme-agnostic.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.sumcheck.jagged.layout import _prepad_folded, _segment_gather_np
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class GkrLayer:
    """One dense fractional-sum layer over (batch || row) variables.

    `num_batch_variables` is the floor: folding stops once the row
    variables are exhausted and only the batch dimension remains. Each
    batch element is one independent LogUp instance -- a consumer may call
    it a lookup *interaction* (the term used throughout this module's
    circuit prose); zorch itself stays scheme-agnostic.
    """

    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array
    num_batch_variables: int

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
        if not 0 <= self.num_batch_variables <= self.num_variables:
            raise ValueError(
                f"num_batch_variables must be in [0, {self.num_variables}], "
                f"got {self.num_batch_variables}"
            )

    @property
    def num_variables(self) -> int:
        return log2_strict_usize(self.numerator_0.shape[0])

    @property
    def num_row_variables(self) -> int:
        return self.num_variables - self.num_batch_variables


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
        num_batch_variables=layer.num_batch_variables,
    )


def build_pyramid(first: GkrLayer) -> list[GkrLayer]:
    """Eager fold from `first` (most row variables) down to the batch floor.

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

    At `num_row_variables == 0` each index is one batch element's fraction with a
    0-child and a 1-child; interleaving recovers the MLE over the batch
    variables plus one.
    """
    if layer.num_row_variables != 0:
        raise ValueError(
            f"extract_outputs expects the batch floor (num_row_variables "
            f"== 0), got {layer.num_row_variables}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )


@dataclass(frozen=True)
class JaggedGkrLayer:
    """One jagged fractional-sum layer, stored batch-major.

    The four MLEs are flat over `sum(row_counts)`: all rows of batch element 0,
    then batch element 1, and so on, so each batch element carries its own row
    count. The batch count must be a power of two (the consumer pads its
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
        log2_strict_usize(self.num_batches)
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
    def num_batches(self) -> int:
        return len(self.row_counts)

    @property
    def num_batch_variables(self) -> int:
        return log2_strict_usize(self.num_batches)

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
    already agree (no gather needed). jnp so the unrolled `_round_metadata`
    schedule rides the jax round body byte-for-byte.
    """
    seg = _segment_gather_np(src_counts, dst_counts)
    return None if seg is None else jnp.asarray(seg)


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


@partial(jax.jit, static_argnames=("row_counts", "out_row_counts"))
def _jagged_transition_core(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    *,
    row_counts: tuple[int, ...],
    out_row_counts: tuple[int, ...],
) -> tuple[Array, Array, Array, Array]:
    """The prepad/fold/postpad numeric core of `jagged_layer_transition`.

    The transition's one `@jit` boundary: eager dispatch decomposes the
    gather/pad/fold into op-by-op kernels, so the core needs exactly one
    boundary around it (`docs/reference/conventions.md` per-island), never zero. A
    consumer that builds the pyramid by iterating the transition per layer --
    rather than the fused `build_jagged_pyramid` -- then pays one fused dispatch
    per transition instead of N eager ops. The schedule is pure static ints, so
    both segment gathers bake into the graph as constants and the static count
    tuples key the compile cache by value: repeated calls at one transition shape
    warm-reuse a single trace, but each DISTINCT shape compiles once -- so
    per-layer iteration recompiles down a deep pyramid, where `build_jagged_pyramid`
    unrolls into one trace regardless of depth.
    """
    prepad_counts, folded_counts = _prepad_folded(row_counts)
    n0, n1, d0, d1 = _pad_neutral(
        numerator_0,
        numerator_1,
        denominator_0,
        denominator_1,
        _segment_gather(row_counts, prepad_counts),
    )
    rn0, rn1, rd0, rd1 = _fold_pairs(n0, n1, d0, d1)
    return _pad_neutral(
        rn0, rn1, rd0, rd1, _segment_gather(folded_counts, out_row_counts)
    )


def jagged_layer_transition(
    layer: JaggedGkrLayer, out_row_counts: tuple[int, ...]
) -> JaggedGkrLayer:
    """Fold one row variable per segment, then pad to the consumer's schedule.

    Odd segments are pre-padded with the additive-identity fraction (n=0, d=1)
    so the flat stride-2 fold never pairs across a batch boundary; the
    folded segments are then padded out to `out_row_counts` with the same
    neutral rows. The schedule is the consumer's policy; this block only
    refuses to truncate, which would silently drop fractions from the sum.

    Host-side validation stays out of the `@jit`; the numeric fold rides
    `_jagged_transition_core`.
    """
    if len(out_row_counts) != layer.num_batches:
        raise ValueError(
            f"schedule must cover all {layer.num_batches} batches, "
            f"got {len(out_row_counts)} entries"
        )
    _, folded_counts = _prepad_folded(layer.row_counts)
    for i, (folded, out) in enumerate(zip(folded_counts, out_row_counts)):
        if out < folded:
            raise ValueError(
                f"schedule truncates batch element {i}: folded row count "
                f"{folded} > target {out}"
            )

    rn0, rn1, rd0, rd1 = _jagged_transition_core(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        row_counts=layer.row_counts,
        out_row_counts=out_row_counts,
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

    The floor is row counts all 1 -- one fraction pair per batch element -- the
    jagged dual of `extract_outputs`'s `num_row_variables == 0` precondition.
    A schedule that stops higher folds the rest down with
    `jagged_layer_transition` first; how far to fold is the consumer's call.
    """
    if any(rc != 1 for rc in layer.row_counts):
        raise ValueError(
            f"extract_jagged_outputs expects the batch floor (row "
            f"counts all 1), got {layer.row_counts}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )


@partial(jax.jit, static_argnames=("first_row_counts", "schedules"))
def _build_pyramid_planes(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    *,
    first_row_counts: tuple[int, ...],
    schedules: tuple[tuple[int, ...], ...],
) -> list[tuple[Array, Array, Array, Array]]:
    """Fold the chain one transition at a time, each layer at its NATURAL width.

    One fused traced region keyed by the static schedule (O(1) dispatches
    regardless of depth), NOT stacked into a uniform width: every layer keeps its
    own geometrically-shrinking height, so peak residency is `sum(natural widths)`
    ~= 2*H rather than `depth * max_width`. The whole pyramid is a required output
    (nothing XLA can rematerialize away), so a uniform-width stack would pin
    `depth * max_width` live at once -- a hard floor under the whole prove.

    Each transition folds at its own shape, so the first layer may enter
    base-field numerators under an extension-field denominator and transition 0's
    fold `n0*d1 + n1*d0` promotes them to EF inline (no separate carve-out).
    """
    planes: list[tuple[Array, Array, Array, Array]] = []
    n0, n1, d0, d1 = numerator_0, numerator_1, denominator_0, denominator_1
    cur = first_row_counts
    for out_row_counts in schedules:
        prepad_counts, folded_counts = _prepad_folded(cur)
        n0, n1, d0, d1 = _pad_neutral(
            n0, n1, d0, d1, _segment_gather(cur, prepad_counts)
        )
        n0, n1, d0, d1 = _fold_pairs(n0, n1, d0, d1)
        n0, n1, d0, d1 = _pad_neutral(
            n0, n1, d0, d1, _segment_gather(folded_counts, out_row_counts)
        )
        planes.append((n0, n1, d0, d1))
        cur = out_row_counts
    return planes


def build_jagged_pyramid(
    first: JaggedGkrLayer, schedules: Sequence[tuple[int, ...]]
) -> list[JaggedGkrLayer]:
    """Build the jagged pyramid `[first, ..., floor]`, folding one row variable
    per transition. `schedules[k]` is transition `k`'s `out_row_counts` (the
    caller's halving policy, the same argument the eager `jagged_layer_transition`
    takes). One fused traced region via `_build_pyramid_planes`, byte-identical to
    iterating `jagged_layer_transition` down the chain, but each layer keeps its
    natural width so peak residency stays ~2*H."""
    schedules = list(schedules)
    if not schedules:
        return [first]

    # Mirror `jagged_layer_transition`'s guards down the chain: a schedule that
    # changes a batch count or truncates a batch element's folded rows would
    # silently drop fractions from the sum (the gather copies only `min(src, dst)`
    # rows), so refuse it loudly here rather than mis-sum.
    cur = first.row_counts
    for out_row_counts in schedules:
        if len(out_row_counts) != len(cur):
            raise ValueError(
                f"schedule must cover all {len(cur)} batches, got "
                f"{len(out_row_counts)} entries"
            )
        _, folded = _prepad_folded(cur)
        for i, (folded_rc, out_rc) in enumerate(zip(folded, out_row_counts)):
            if out_rc < folded_rc:
                raise ValueError(
                    f"schedule truncates batch element {i}: folded row count "
                    f"{folded_rc} > target {out_rc}"
                )
        cur = out_row_counts

    planes = _build_pyramid_planes(
        first.numerator_0,
        first.numerator_1,
        first.denominator_0,
        first.denominator_1,
        first_row_counts=first.row_counts,
        schedules=tuple(schedules),
    )
    layers = [first]
    for out_row_counts, (n0, n1, d0, d1) in zip(schedules, planes):
        layers.append(
            JaggedGkrLayer(
                numerator_0=n0,
                numerator_1=n1,
                denominator_0=d0,
                denominator_1=d1,
                row_counts=out_row_counts,
            )
        )
    return layers
