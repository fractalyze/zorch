# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Proof-of-work grind search: the lowest counter whose candidate passes.

One engine for every transcript's grind. Each `lax.while_loop` step tests a
`window`-wide counter batch IN PARALLEL through the caller-supplied predicate
and keeps the lowest-index hit; the loop tiles windows only because the full
space cannot be tested at once, and it early-exits at the first window that
hits — for a typical difficulty the hit is in the first window, so the loop
runs once. What "candidate" means is the caller's: the algebraic duplex checks
the challenge its observe+sample induces, a byte-framed transcript checks the
digest of a pre-image built from `state_digest` and the counter. On exhaustion
the trailing fallback 0 is returned unchecked — the caller's verify/check-witness
is the soundness gate, so which counter the search returns is soundness-neutral
(the `DuplexTranscript.grind` contract).
"""
from __future__ import annotations

from collections.abc import Callable

import frx.numpy as fnp
from frx import Array, lax

# Counters a while_loop step tests in parallel. A window is an upper bound on
# the batch, not the batch: `grind_window_for` sizes it to the difficulty, and
# a search never tests more than this many at once.
GRIND_WINDOW = 1 << 16

# Floor on the window, so an easy search still arrives as one wide device batch
# rather than a launch-bound sliver. Below roughly this width the kernel is
# latency-bound and a narrower batch buys nothing back.
MIN_GRIND_WINDOW = 1 << 10


def grind_window_for(pow_bits: int) -> int:
    """Counters to test per step for a `pow_bits` search.

    The window is a work/launch trade, never a correctness parameter:
    `grind_search` scans windows in increasing order and takes the lowest hit
    inside one, so it returns the same counter at any width. Sizing it to the
    difficulty is therefore free of protocol consequence.

    A search costs at least one full window however easy it is, so a fixed
    `GRIND_WINDOW` makes every low-difficulty grind pay for 2^16 evaluations to
    find a hit expected within 2^bits. Two windows' worth of expected work
    finds one ~86% of the time, and a miss just runs the loop again."""
    if pow_bits < 0:
        raise ValueError(f"pow_bits must be >= 0, got {pow_bits}")
    return min(GRIND_WINDOW, max(MIN_GRIND_WINDOW, 1 << (pow_bits + 1)))


def leading_zero_bits_ok(digests: Array, bits: int) -> Array:
    """Whether each digest (uint8 `[B, digest_size]`) has >= `bits` leading zero
    bits, big-endian (digest[..., 0] most significant). Lives here for the same
    reason the search does: it is the predicate every byte-framed transcript's
    `check_batch` ends in, whatever hash built the digest. Traceable, so it
    composes into the search's one device program; byte-identical to
    `byte_transcript._leading_zero_bits_ok`, its host twin."""
    full, extra = divmod(bits, 8)
    ok = fnp.all(digests[:, :full] == 0, axis=1)
    if extra:
        # Weakly-typed literal: a uint8 shift operand would be a device array.
        ok = ok & ((digests[:, full] >> (8 - extra)) == 0)
    return ok


def grind_search(
    check_batch: Callable[[Array], Array], bound: int, window: int = GRIND_WINDOW
) -> Array:
    """Lowest uint32 counter in `[0, bound)` for which `check_batch` reports a
    hit. `check_batch`: uint32 `[window]` counters -> bool `[window]`, pure and
    traceable — the search is jit-composable and runs as one device program.
    `bound` caps `base` so `base + window` cannot wrap uint32."""
    offsets = fnp.arange(window, dtype=fnp.uint32)
    bound_u32 = fnp.uint32(min(bound, 2**32 - window))

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        found, base, _ = carry
        return fnp.logical_and(fnp.logical_not(found), base < bound_u32)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        found, base, best = carry
        hits = check_batch(base + offsets)
        first = fnp.min(fnp.where(hits, offsets, fnp.uint32(window)))
        any_hit = first < fnp.uint32(window)
        return (
            fnp.logical_or(found, any_hit),
            base + fnp.uint32(window),
            fnp.where(any_hit, base + first, best),
        )

    init = (fnp.bool_(False), fnp.uint32(0), fnp.uint32(0))
    _found, _base, counter = lax.while_loop(cond, body, init)
    return counter
