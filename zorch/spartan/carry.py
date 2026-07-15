# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The Spartan chain's accumulating seam carry
(`docs/composition/stage-composition.md`, "The pipeline carry").

One frozen pytree threaded through the chain, one field per seam-crossing value;
each stage `replace`s its own fields and leaves a field `None` until its writer
runs. A value that skips a stage — `r_x`, read by both the inner sumcheck and the
PCS glue — rides here rather than through a pass-through pairwise seam type.
"""

from __future__ import annotations

from dataclasses import dataclass

from frx import Array
from frx.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class SpartanCarry:
    # Written by the outer (zerocheck) stage; read by the inner and glue stages.
    r_x: Array | None = None  # outer sumcheck point (s_x coords)
    claims_outer: Array | None = None  # (Az, Bz, Cz)(r_x)
    # Written by the RLC combinator; read by the inner and glue stages.
    r_batch: Array | None = None  # batching challenge r
    joint_claim: Array | None = None  # Az + r·Bz + r²·Cz
    # Written by the inner (lincheck) stage; read by the glue stage.
    r_y: Array | None = None  # inner sumcheck point (s_y coords)
    inner_final: Array | None = None  # reduced inner claim = eval_ABC · z̃(r_y)


def _require(value: Array | None, field: str, writer: str) -> Array:
    """Read a carry field, failing loud if its writer stage has not run — a
    mis-sequenced chain is a construction bug caught on the first call."""
    if value is None:
        raise ValueError(f"carry.{field} is unset; the {writer} stage must run first")
    return value
