# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-round Fiat-Shamir hop + scalar reduce: observe the round
poly, squeeze the challenge, fold the claim/pad scalars, and slice the
next eval-point coordinate -- traced into the whole-layer jit, one fused
region per round."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
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
    eval_point: Array,
    pos: Array,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, Array]:
    """Reinterpret the squeezed challenge, fold the round scalars, AND slice the
    next round's eval-point coordinate. Three hops collapse here: the challenge
    reshape/bitcast, the scalar fold, and the per-round `eval_point` gather (a
    `jnp.take` is a real ~22us dispatch, NOT a buffer view). `pos` indexes this
    round's coordinate; the next is `pos - 1`, threaded device-resident so no
    per-round index round-trips the host. Returns the round challenge `r`, the next
    `claim`, `pad_adj`, the next round's `z_cur`, and the decremented `pos`. Plain
    (un-jitted) so it fuses into whichever kernel owns it -- the round loop's
    `_fs_reduce`, which prepends the Fiat-Shamir hop."""
    r = reinterpret_challenge(raw, dtype)
    claim, pad_adj = gruen.fold_round_scalars(poly, r, pad_adj, z_cur)
    # The last round's `pos_next` is -1 (a dead output -- no round consumes it);
    # clamp so the slice index is provably in-bounds rather than leaning on
    # `dynamic_slice`'s implicit index clamp. No-op for every live round (pos >= 1).
    pos_next = jnp.maximum(pos - 1, jnp.int32(0))
    z_next = jax.lax.dynamic_index_in_dim(eval_point, pos_next, keepdims=False)
    return r, claim, pad_adj, z_next, pos_next


def _fs_reduce(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    eval_point: Array,
    pos: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array, Array, Array]:
    """The per-round FS hop + reduce: observe `poly`, squeeze the challenge, then
    `_reduce_body`. Returns the advanced transcript and `(r, claim, pad_adj, z_next,
    pos_next)`. No jit of its own -- it fuses into the round's compute under the
    whole-layer jit.

    The device FS hop rides the `zorch.duplex_fs` composite
    (`observe_and_sample_marked`) so the whole absorb+squeeze lowers to ONE
    register-resident kernel. Without the marker the duplex glue (rate-block merge,
    position select, output extract) decomposes into ~6k loop-fused ops/hop,
    dominating the layer compile; the generic fused_region path is declined by the
    vendor (exponential LoopFusion), so the dedicated `zorch.duplex_fs` emitter is
    what fuses it."""
    transcript, raw = observe_and_sample_marked(transcript, poly, n)
    r, claim, pad_adj, z_next, pos_next = _reduce_body(
        raw, poly, pad_adj, z_cur, eval_point, pos, dtype
    )
    return transcript, r, claim, pad_adj, z_next, pos_next
