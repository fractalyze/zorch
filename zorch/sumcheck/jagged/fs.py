# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-round Fiat-Shamir hop + scalar reduce: observe the round
poly, squeeze the challenge, and fold the claim/pad scalars -- traced
into the whole-layer jit, one fused region per round. (The round's
eval-point coordinate is sliced statically by the caller's loop index.)

The reduce (`eval_coeffs(poly, r)` + `mass * eq_factor(r, z)`) stays plain
device ops that consume the `zorch.duplex_fs` composite's challenge output --
zorch stays optimization-agnostic, leaving any FS-kernel absorption to the
consumer's xla layer rather than baking it in here."""

from __future__ import annotations

from typing import Any

from jax import Array

from zorch.sumcheck import gruen
from zorch.transcript import (
    DuplexTranscript,
    observe_and_sample_marked,
    reinterpret_challenge,
)


def _reduce_body(
    raw: Array,
    poly: Array,
    pad_adj: Array,
    z_cur: Array,
    dtype: Any,
) -> tuple[Array, Array, Array]:
    """Reinterpret the squeezed challenge and fold the round scalars. The round's
    `eval_point` coordinate `z_cur` is sliced statically by the caller (the loop
    index is a compile-time constant), so no per-round gather rides here -- a
    device-resident index would cost a ~22us `jnp.take` dispatch every round.
    Returns the round challenge `r`, the next `claim`, and `pad_adj`. Plain
    (un-jitted) so it fuses into whichever kernel owns it -- the round loop's
    `_fs_reduce`."""
    r = reinterpret_challenge(raw, dtype)
    claim, pad_adj = gruen.fold_round_scalars(poly, r, pad_adj, z_cur)
    return r, claim, pad_adj


def _fs_reduce(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array]:
    """The per-round FS hop + reduce: observe `poly`, squeeze the challenge, then
    `_reduce_body`. Returns the advanced transcript and `(r, claim, pad_adj)`. No
    jit of its own -- it fuses into the round's compute under the whole-layer jit.

    The device FS hop rides the `zorch.duplex_fs` composite
    (`observe_and_sample_marked`) so the whole absorb+squeeze lowers to ONE
    register-resident kernel. The reduce that consumes its challenge stays plain
    device ops -- optimization-agnostic, left for the consumer's xla layer."""
    transcript, raw = observe_and_sample_marked(transcript, poly, n)
    r, claim, pad_adj = _reduce_body(raw, poly, pad_adj, z_cur, dtype)
    return transcript, r, claim, pad_adj
