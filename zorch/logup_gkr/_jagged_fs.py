# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-round Fiat-Shamir hop + scalar reduce: observe the round
poly, squeeze the challenge, fold the claim/pad scalars, and slice the
next eval-point coordinate -- one fused zone per hop on the eager path."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.poly.univariate import (
    eval_coeffs,
)
from zorch.transcript import (
    DuplexTranscript,
    observe_and_sample_marked,
    reinterpret_challenge,
)


def _fold_scalars(
    poly: Array, r: Array, pad_adj: Array, z: Array, one: Array
) -> tuple[Array, Array]:
    """The per-round scalar fold: the next claim (round poly evaluated at `r`) and the
    updated pad-mass `pad_adj`. One source for both the oracle
    `_run_jagged_rounds_reference` (which inlines it) and the round loop's
    `_reduce_body`, so the two cannot drift out of byte-equality."""
    return eval_coeffs(poly, r), pad_adj * (z * r + (one - z) * (one - r))


def _reduce_body(
    raw: Array,
    poly: Array,
    pad_adj: Array,
    z_cur: Array,
    one: Array,
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
    claim, pad_adj = _fold_scalars(poly, r, pad_adj, z_cur, one)
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
    whole-layer jit. `one` is baked.

    The device FS hop rides the `zorch.duplex_fs` composite
    (`observe_and_sample_marked`) so the whole absorb+squeeze lowers to ONE
    register-resident kernel. Without the marker the duplex glue (rate-block merge,
    position select, output extract) decomposes into ~6k loop-fused ops/hop,
    dominating the layer compile; the generic fused_region path is declined by the
    vendor (exponential LoopFusion), so the dedicated `zorch.duplex_fs` emitter is
    what fuses it."""
    transcript, raw = observe_and_sample_marked(transcript, poly, n)
    one = jnp.ones((), dtype)
    r, claim, pad_adj, z_next, pos_next = _reduce_body(
        raw, poly, pad_adj, z_cur, one, eval_point, pos, dtype
    )
    return transcript, r, claim, pad_adj, z_next, pos_next


# Fixed eval_point width for the FS-hop zone below: the recognizer bounds
# num_vars at 62, so 64 covers every layer, and one padded width keeps the
# zone's compile key layer-invariant (the pad tail is never read — the
# dynamic_index rides `pos`, which always points into the live prefix).
_FS_EVAL_POINT_CAP = 64

# The export path's per-round FS hop, hoisted into a module-level jit zone (the
# `_composite.py`-recommended pattern, mirroring poseidon2's `_permute_body`):
# called eagerly between round binaries, the bare `_fs_reduce` re-traces the
# `zorch.duplex_fs` composite's Python body EVERY round and enqueues
# `_reduce_body`'s ~15 element ops one dispatch at a time — measured as the
# dominant host wall of the warm decoupled prove (~330 rounds/shard). The zone
# collapses each hop to one cached-executable dispatch. Keyed by operand
# shapes (eval_point length varies per layer) plus the static squeeze count and
# dtype. jit is byte-transparent, so the transcript stream is unchanged.
_fs_reduce_zone = partial(jax.jit, static_argnums=(6, 7))(_fs_reduce)


def _fs_reduce_dispatch(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    eval_point: Array,
    pos: Array,
    n: int,
    dtype: Any,
) -> tuple[DuplexTranscript, Array, Array, Array, Array, Array]:
    """Route the FS hop through the jit zone on the eager/export path; under an
    outer trace call the plain body so it keeps fusing into the whole-layer
    program exactly as before (the zone would inline there anyway, but staying
    out preserves the traced path's structure byte-for-byte by construction)."""
    if isinstance(poly, jax.core.Tracer):
        return _fs_reduce(poly, transcript, pad_adj, z_cur, eval_point, pos, n, dtype)
    return _fs_reduce_zone(poly, transcript, pad_adj, z_cur, eval_point, pos, n, dtype)
