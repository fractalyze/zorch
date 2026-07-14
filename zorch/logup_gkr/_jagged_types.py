# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-specific dataclasses of the jagged LogUp-GKR prover: the four MLE
planes, the per-round scalar bundle, and the round-loop state. The
scheme-agnostic caps/schedule/interp-consts live in
`zorch.sumcheck.jagged.types` (`jagged_prover` has the protocol overview)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
from frx import Array

# eq (deg 1) * (lam*(n0*d1 + n1*d0) + d0*d1) (deg 2), in coefficient form.
_DEGREE = 3


@partial(
    frx.tree_util.register_dataclass,
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
    frx.tree_util.register_dataclass,
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


@partial(
    frx.tree_util.register_dataclass,
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
