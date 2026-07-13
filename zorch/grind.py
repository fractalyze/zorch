# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Proof-of-work grind search: the lowest counter whose candidate passes.

One engine for every transcript's grind. Each `lax.while_loop` step tests a
`window`-wide counter batch IN PARALLEL through the caller-supplied predicate
and keeps the lowest-index hit; the loop tiles windows only because the full
space cannot be tested at once, and it early-exits at the first window that
hits — for a typical difficulty the hit is in the first window, so the loop
runs once. What "candidate" means is the caller's: the algebraic duplex checks
the challenge its observe+sample induces, the SHA-256 transcript checks the
digest of `state_digest || counter_le8`. On exhaustion the trailing fallback 0
is returned unchecked — the caller's verify/check-witness is the soundness
gate, so which counter the search returns is soundness-neutral (the
`DuplexTranscript.grind` contract).
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

# Counters a while_loop step tests in parallel: one window covers any practical
# difficulty (expected work 2^bits), so the loop normally runs once.
GRIND_WINDOW = 1 << 16


def grind_search(
    check_batch: Callable[[Array], Array], bound: int, window: int = GRIND_WINDOW
) -> Array:
    """Lowest uint32 counter in `[0, bound)` for which `check_batch` reports a
    hit. `check_batch`: uint32 `[window]` counters -> bool `[window]`, pure and
    traceable — the search is jit-composable and runs as one device program.
    `bound` caps `base` so `base + window` cannot wrap uint32."""
    offsets = jnp.arange(window, dtype=jnp.uint32)
    bound_u32 = jnp.uint32(min(bound, 2**32 - window))

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        found, base, _ = carry
        return jnp.logical_and(jnp.logical_not(found), base < bound_u32)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        found, base, best = carry
        hits = check_batch(base + offsets)
        first = jnp.min(jnp.where(hits, offsets, jnp.uint32(window)))
        any_hit = first < jnp.uint32(window)
        return (
            jnp.logical_or(found, any_hit),
            base + jnp.uint32(window),
            jnp.where(any_hit, base + first, best),
        )

    init = (jnp.bool_(False), jnp.uint32(0), jnp.uint32(0))
    _found, _base, counter = lax.while_loop(cond, body, init)
    return counter
