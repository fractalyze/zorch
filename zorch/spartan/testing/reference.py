# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""An independent, brute-force Spartan reference for structural cross-checking.

The production combinators fold the sumcheck with `StandardRound`
(`summand_evals` / `domain.sample` / `vmap`); this reference recomputes the same
round-polynomial and claimed-eval sequences by a *different* code path — an
explicit per-point, per-round loop over the hypercube — so matching them
validates the production path implements the protocol, not that it agrees with
itself. The matched objects (degrees, per-round eval tuples, running claims,
powers-of-`r` batching, final evals) are the field-agnostic algebraic skeleton a
cross-field comparison can share.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import frx.numpy as jnp
from frx import Array

from zorch.transcript import Transcript


def naive_round_polys(
    tables: Sequence[Array],
    combine: Callable[..., Array],
    degree: int,
    challenges: Sequence[Array],
) -> tuple[list[Array], list[Array]]:
    """Brute-force natural-domain round polynomials for `Σ_x combine(f₁…)(x)`.

    Binds MSB-first (contiguous halves), one variable per challenge. Each round
    poly is `[s(0), …, s(degree)]` with `s(u) = Σ_{x'} combine(*folded_at_u)`,
    computed by an explicit fold at each integer point `u` — independent of the
    production `summand_evals` path. Returns `(round_polys, final_tables)`.
    """
    dtype = tables[0].dtype
    tabs = [jnp.asarray(t) for t in tables]
    polys: list[Array] = []
    for r in challenges:
        half = tabs[0].shape[0] // 2
        pts = []
        for u in range(degree + 1):
            uf = jnp.asarray(u, dtype)
            folded = [t[:half] + uf * (t[half:] - t[:half]) for t in tabs]
            pts.append(jnp.sum(combine(*folded)))
        polys.append(jnp.stack(pts))
        tabs = [t[:half] + r * (t[half:] - t[:half]) for t in tabs]
    return polys, tabs


def replay_challenges(
    transcript: Transcript,
    commitment: Array,
    io: Array,
    outer_polys: Array,
    claims: Array,
    inner_polys: Array,
    s_x: int,
) -> dict[str, Array]:
    """Re-derive `(τ, r_x, r_batch, r_y)` by replaying the assembly's exact
    Fiat-Shamir schedule against the proof messages."""
    t = transcript.observe(commitment)
    if io.shape[0] > 0:
        t = t.observe(io)
    t, tau = t.sample(s_x)
    r_x = []
    for msg in outer_polys:
        t, r = t.observe_and_sample(msg, 1)
        r_x.append(r[0])
    t = t.observe(claims)
    t, rb = t.sample(1)
    r_batch = rb[0]
    r_y = []
    for msg in inner_polys:
        t, r = t.observe_and_sample(msg, 1)
        r_y.append(r[0])
    return {
        "tau": tau,
        "r_x": jnp.stack(r_x),
        "r_batch": r_batch,
        "r_y": jnp.stack(r_y),
    }
