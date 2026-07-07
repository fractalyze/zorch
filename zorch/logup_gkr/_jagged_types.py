# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared dataclasses of the jagged LogUp-GKR prover: the proof
wire type, the fixed-width round caps, and the round-loop state/schedule
bundles (`jagged_prover` has the protocol overview)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
from jax import Array

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2), in coefficient form.
_DEGREE = 3


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


# ===== per-round jagged sumcheck engine =====
# The per-round compute kernels + the host loop that threads them; each round's
# compute + device Fiat-Shamir hop traces into the whole-layer jit.


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["n0", "n1", "d0", "d1"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _Planes:
    """The four LogUp MLE planes (numerator_0/1, denominator_0/1) as one pytree --
    they travel and bind together through every round. A registered pytree so it
    crosses the whole-layer jit boundary as a single structured operand."""

    n0: Array
    n1: Array
    d0: Array
    d1: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["eq_adj", "pad_adj", "z_cur", "claim", "lam"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _RoundScalars:
    """The per-round scalar inputs to the round univariate: the eq/pad bound-mass
    corrections (`eq_adj`/`pad_adj`), the current point coordinate `z_cur`, the
    running `claim`, and the LogUp RLC coefficient `lam`. Scalar operands of the
    exported round kernel (a registered pytree)."""

    eq_adj: Array
    pad_adj: Array
    z_cur: Array
    claim: Array
    lam: Array


@dataclass(frozen=True)
class _InterpConsts:
    """The Lagrange interpolation constants (the `{0..DEGREE}` natural domain and
    the inverse Vandermonde). They depend only on dtype, so the round kernels bake
    them in as closure constants -- NOT export operands, so not a pytree."""

    naturals: Array
    inv_vand: Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["planes", "eq_row", "eq_int", "eval_point", "lam", "claim"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _JaggedState:
    """A jagged layer's sumcheck carry: the four MLE planes, the row/batch
    eq tables, the bound point, the RLC `lam`, and the running `claim`. Bundled so
    the round-loop functions take one state instead of nine loose arrays -- the
    `(round, state, transcript)` shape `sumcheck.prove` and jagged-pcs's
    `_InnerState` already use."""

    planes: _Planes
    eq_row: Array
    eq_int: Array
    eval_point: Array
    lam: Array
    claim: Array


@dataclass(frozen=True)
class _JaggedSchedule:
    """A jagged layer's static round schedule: the layer's `row_counts`
    operand plus the per-round live triples (marker v2 — the re-pad schedule
    itself derives in-kernel from these), the interpolation constants, and
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
