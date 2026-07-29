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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, partial

import frx
import frx.numpy as fnp
from frx import Array

from zorch._composite import composite
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    SUMCHECK_ROUND_MARKER_VERSION,
)
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
    return fnp.stack([child_0, child_1], axis=-1).flatten()


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


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "numerator_0",
        "numerator_1",
        "denominator_0",
        "denominator_1",
        "row_counts",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class JaggedGkrLayer:
    """One jagged fractional-sum layer at a capacity width, stored batch-major.

    The four MLEs are flat over a static capacity `width >= sum(row_counts)`:
    the live prefix holds all rows of batch element 0, then batch element 1,
    and so on; every slot past `sum(row_counts)` is DEAD and laid zero. The
    counts ride as one traced i32[num_batches] vector, so no per-input layout
    value keys a compile — transitions derive their gathers in-trace and the
    layer rounds take the schedule as traced operands, keying on the capacity
    shapes alone. An exactly-sized layer (`width == sum(row_counts)`) is just
    the zero-slack case; the batch count must be a power of two (the consumer
    pads its interaction list).

    The dead tail is never read: a transition resolves every non-live source
    through its gather sentinel, and the layer rounds bound their reads by
    the live-pair operand. Zero (not the neutral fraction) so a reduction
    that sweeps a fixed-width buffer picks up nothing from the dead region.
    Truncation-safety is likewise the consumer's obligation — a host guard
    cannot read the traced counts, so a transition schedule must dominate
    `ceil(rc / 2)` pointwise and its width must hold its count sum for every
    input the consumer admits.
    """

    numerator_0: Array
    numerator_1: Array
    denominator_0: Array
    denominator_1: Array
    row_counts: Array

    def __post_init__(self) -> None:
        # Shape-only checks: `register_dataclass` reruns this during
        # unflatten, so no value may be branched on here — and the leaves are
        # not always arrays: AOT lowering (`jit(f).lower(layer)`) rebuilds the
        # tree with `frx.stages.ArgInfo` leaves, which expose `shape`/`dtype`
        # but no `ndim`. Stick to `.shape`.
        log2_strict_usize(self.num_batches)
        if len(self.row_counts.shape) != 1:
            raise ValueError(
                f"row_counts must be a flat vector, got {self.row_counts.shape}"
            )
        width = self.width
        for name in (
            "numerator_0",
            "numerator_1",
            "denominator_0",
            "denominator_1",
        ):
            shape = getattr(self, name).shape
            if shape != (width,):
                raise ValueError(
                    f"the four MLEs must share one capacity width; "
                    f"{name} is {shape}, expected ({width},)"
                )

    @property
    def num_batches(self) -> int:
        return self.row_counts.shape[0]

    @property
    def num_batch_variables(self) -> int:
        return log2_strict_usize(self.num_batches)

    @property
    def width(self) -> int:
        return self.numerator_0.shape[0]


@cache
def _counts_operand(row_counts: tuple[int, ...]) -> Array:
    """A host-known count tuple as its committed i32 device vector — one
    `device_put` per distinct layout (the floor schedules re-commit per
    prove otherwise). Concrete via `ensure_compile_time_eval` so the cached
    value never escapes a trace as a tracer."""
    with frx.ensure_compile_time_eval():
        return frx.device_put(fnp.asarray(row_counts, fnp.int32))


def _gather_pad(arr: Array, gather: Array, pad_val: int) -> Array:
    """Gather with sentinel-aware padding.

    Clamped take + where rather than appending the padding value to `arr`,
    which would copy the whole array.
    """
    is_pad = gather >= arr.shape[0]
    clamped = fnp.minimum(gather, arr.shape[0] - 1)
    return fnp.where(is_pad, pad_val, arr[clamped])


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


def _derive_transition_gathers(
    row_counts: Array, out_row_counts: Array, out_width: int, sentinel: int
) -> tuple[Array, Array, Array]:
    """The transition's two `_segment_gather` hops and the stride-2 pairing
    composed into per-output-slot (even, odd) source indices plus the
    out-layout live mask, derived in-trace from the traced counts -- the
    transition twin of the round side's `_derive_row_schedule`.

    Output slot `q` lands in segment `s` (searchsorted over the out layout's
    cumsum) at in-segment index `j`; a live `j < ceil(rc[s] / 2)` folds
    prepadded elements `(2j, 2j+1)` at the segment's traced base offset. The
    odd element of an odd segment's last pair and every in-layout slot past a
    segment's folded count resolve to `sentinel`, which `_gather_pad` maps to
    the neutral fraction -- the same values the host-built gathers bake in as
    constants, so the fold multiplies identical operands either way. The
    returned `live` marks slots inside `sum(out_row_counts)`; everything past
    it is the dead region the core zeroes.
    """
    i32 = fnp.int32
    rc = row_counts.astype(i32)
    orc = out_row_counts.astype(i32)
    folded = rc + i32(1) >> 1
    base = fnp.concatenate([fnp.zeros((1,), i32), fnp.cumsum(rc)[:-1]])
    cum_out = fnp.cumsum(orc)
    q = fnp.arange(out_width, dtype=i32)
    # scan_unrolled: the default scan method lowers to a while whose carry is
    # the full index array, an every-trip device copy; unrolling the
    # log2(num_batches) steps is byte-identical.
    s = fnp.searchsorted(cum_out, q, side="right", method="scan_unrolled")
    s = fnp.minimum(s.astype(i32), i32(row_counts.shape[0] - 1))
    j = q - (cum_out[s] - orc[s])
    live = q < cum_out[-1]
    folds = live & (j < folded[s])
    src_even = base[s] + j * i32(2)
    sent = i32(sentinel)
    gather_even = fnp.where(folds, src_even, sent)
    gather_odd = fnp.where(
        folds & (j * i32(2) + i32(1) < rc[s]), src_even + i32(1), sent
    )
    return gather_even, gather_odd, live


def _transition_composite_decomp(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    row_counts: Array,
    out_row_counts: Array,
    *,
    out_width: int,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array]:
    """The `zorch.sumcheck.round` (variant=transition) decomposition — the
    byte-exact fallback a recognizing emitter replaces: derive the (even, odd)
    gathers in-trace from the traced counts, apply `_fold_pairs`' algebra to
    the gathered operands, and zero the dead region past the live out rows."""
    gather_even, gather_odd, live = _derive_transition_gathers(
        row_counts, out_row_counts, out_width, numerator_0.shape[0]
    )
    n0e = _gather_pad(numerator_0, gather_even, 0)
    n1e = _gather_pad(numerator_1, gather_even, 0)
    d0e = _gather_pad(denominator_0, gather_even, 1)
    d1e = _gather_pad(denominator_1, gather_even, 1)
    n0o = _gather_pad(numerator_0, gather_odd, 0)
    n1o = _gather_pad(numerator_1, gather_odd, 0)
    d0o = _gather_pad(denominator_0, gather_odd, 1)
    d1o = _gather_pad(denominator_1, gather_odd, 1)

    def dead_zeroed(val: Array) -> Array:
        # The fold may promote a base-field numerator to the denominator's
        # extension field, so the zero must take the FOLDED dtype.
        return fnp.where(live, val, fnp.zeros((), dtype=val.dtype))

    return (
        dead_zeroed(n0e * d1e + n1e * d0e),
        dead_zeroed(n0o * d1o + n1o * d0o),
        dead_zeroed(d0e * d1e),
        dead_zeroed(d0o * d1o),
    )


@partial(frx.jit, static_argnames=("out_width",))
def _jagged_transition_core(
    numerator_0: Array,
    numerator_1: Array,
    denominator_0: Array,
    denominator_1: Array,
    row_counts: Array,
    out_row_counts: Array,
    *,
    out_width: int,
) -> tuple[Array, Array, Array, Array]:
    """The transition's one `@jit` boundary, marked with the
    `zorch.sumcheck.round` transition variant: a recognizing emitter fuses the
    whole map into one kernel (the gather graph materializes its
    intermediates); an unclaiming compiler runs the decomposition inline,
    byte-identical. The compile keys on (input width, output width, batch
    count, dtypes) alone -- capacity constants, never one input's layout."""
    return composite(
        partial(_transition_composite_decomp, out_width=out_width),
        numerator_0,
        numerator_1,
        denominator_0,
        denominator_1,
        row_counts,
        out_row_counts,
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        variant="transition",
    )


def jagged_layer_transition(
    layer: JaggedGkrLayer,
    out_row_counts: Array | Sequence[int],
    out_width: int | None = None,
) -> JaggedGkrLayer:
    """Fold one row variable per segment into a fresh `out_width` capacity.

    Odd segments pre-pad with the additive-identity fraction (n=0, d=1) so
    the stride-2 fold never pairs across a batch boundary; slots between a
    segment's folded count and its out count are live neutral padding; the
    dead region past `sum(out_row_counts)` lands zero.

    `out_row_counts` is the consumer's halving policy: a host sequence for a
    statically-known schedule (`out_width` defaults to its sum — the
    zero-slack layout), or the policy evaluated on ITS traced counts, with
    `out_width` the static capacity holding it. A schedule that truncates a
    segment's folded rows cannot be rejected host-side (the counts are
    traced), so the consumer owns that guarantee — its policy must dominate
    `ceil(rc / 2)` pointwise for every input it admits.
    """
    if isinstance(out_row_counts, Array):
        if out_width is None:
            raise ValueError("a traced schedule needs an explicit out_width capacity")
        counts = out_row_counts
        num_out = counts.shape[0]
    else:
        host = tuple(int(rc) for rc in out_row_counts)
        counts = _counts_operand(host)
        num_out = len(host)
        if out_width is None:
            out_width = sum(host)
    if num_out != layer.num_batches:
        raise ValueError(
            f"schedule must cover all {layer.num_batches} batches, got "
            f"{num_out} entries"
        )
    rn0, rn1, rd0, rd1 = _jagged_transition_core(
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
        layer.row_counts,
        counts,
        out_width=out_width,
    )
    return JaggedGkrLayer(
        numerator_0=rn0,
        numerator_1=rn1,
        denominator_0=rd0,
        denominator_1=rd1,
        row_counts=counts,
    )


def build_jagged_pyramid(
    first: JaggedGkrLayer,
    schedules: Sequence[tuple[Array, int] | Sequence[int]],
) -> list[JaggedGkrLayer]:
    """Build the jagged pyramid `[first, ..., floor]`, folding one row
    variable per transition. `schedules[k]` is transition `k`'s policy —
    `(out_row_counts, out_width)` for a traced schedule, or a bare host
    sequence at its zero-slack width — the same argument
    `jagged_layer_transition` takes. Each transition is its own dispatch, so
    the layers land as separate per-layer buffers. Peak residency is ~2*H --
    every natural-width layer is a required GKR input, live until its
    top-down sumcheck -- split across ~depth buffers a pooling allocator can
    seat individually; a single contiguous 2*H alloc exceeds what BFC can
    place on wide shards (#468). One compile per distinct
    (in_width, out_width) pair, shared by every input of the capacity class.
    """
    layers = [first]
    layer = first
    for schedule in schedules:
        if (
            isinstance(schedule, tuple)
            and len(schedule) == 2
            and isinstance(schedule[0], Array)
        ):
            layer = jagged_layer_transition(layer, schedule[0], schedule[1])
        else:
            layer = jagged_layer_transition(layer, schedule)
        layers.append(layer)
    return layers


def extract_jagged_outputs(layer: JaggedGkrLayer) -> LogUpGkrOutput:
    """Interleave the floor layer's children into the output MLEs.

    The floor is row counts all 1 -- one fraction pair per batch element --
    the jagged dual of `extract_outputs`'s `num_row_variables == 0`
    precondition. Row counts are traced, so the gate is the static dual the
    layout implies: a fully-live all-ones layer is exactly `width ==
    num_batches` (any live count above 1 would need more width; a dead slot
    would mean a count of 0, which no saturating fold produces). A schedule
    that stops higher folds the rest down with `jagged_layer_transition`
    first; how far to fold is the consumer's call.
    """
    if layer.width != layer.num_batches:
        raise ValueError(
            f"extract_jagged_outputs expects the batch floor (width == "
            f"num_batches == {layer.num_batches}), got width {layer.width}"
        )
    return LogUpGkrOutput(
        numerator=_interleave(layer.numerator_0, layer.numerator_1),
        denominator=_interleave(layer.denominator_0, layer.denominator_1),
    )
