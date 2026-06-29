# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-only kernels behind the IPA fold — no EC point, no MSM.

These are the pieces a verifier (and an accumulation consumer) needs that touch
*only* the scalar field, factored out of `verifier.py` so they run and test on a
CPU-friendly base field independent of the bn254 `lax.msm` path (the same split
KZG draws between `_quotient_and_eval` and its MSMs). Three things live here:

- `inner_powers` — the evaluation vector `b = (1, x, …, x^{n-1})` IPA proves the
  inner product against (`⟨a, b⟩ = p(x)`).
- `challenge_vector` — the size-`n` vector `s` with `G_final = ⟨s, G⟩` (the one
  expensive MSM the verifier/decider owes). Built by the *exact* inverse of the
  prover's basis fold, so `⟨s, G⟩` reproduces the prover's collapsed basis by
  construction rather than by a re-derived closed form.
- `eval_challenge_poly` — `g(x) = ∏_j (u_j⁻¹ + u_j · x^{2^{k-1-j}})`, the O(log n)
  evaluation of the challenge polynomial whose coefficients are `s`. This is the
  folded scalar `b` *without* materializing `s`, and the reason an accumulation
  step stays succinct: `g` is pinned by the `k = log n` challenges alone (see
  docs/pcs.md and the accumulation-zorch study note §1.3).

`challenge_vector` and `eval_challenge_poly` are two readings of the *same* object
— `eval_challenge_poly(u, x) == ⟨challenge_vector(u), inner_powers(x, n)⟩` — and a
test pins that identity so the succinct path and the explicit path cannot drift.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _check_pow2(n: int) -> int:
    """Return `k` with `n == 2^k`, or raise. IPA folds in half each round, so a
    non-power-of-two length has no last round that collapses to a scalar."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"IPA needs a power-of-two length, got {n}")
    return n.bit_length() - 1


def inner_powers(x: Array, n: int) -> Array:
    """`b = (1, x, x², …, x^{n-1})` as an ascending array. The vector IPA opens
    the inner product against: `⟨a, b⟩ = Σ aᵢ xⁱ = p(x)`. The power chain is
    unrolled at trace time (`n` static) rather than a `jnp.cumprod`: the repo
    keeps field reductions on `lax`/unrolled forms, not the `jnp` reduce wrappers
    (zorch/fusion.py), and the same idiom drives KZG's `_quotient_and_eval`."""
    _check_pow2(n)
    powers = [jnp.ones((), dtype=x.dtype)]
    for _ in range(n - 1):
        powers.append(powers[-1] * x)
    return jnp.stack(powers)


def challenge_vector(u: Array, u_inv: Array) -> Array:
    """The size-`n` combiner `s` with `G_final = ⟨s, G⟩` and `b_final = ⟨s, b⟩`,
    where `u[j]` is round `j`'s challenge and `u_inv[j]` its inverse (`n = 2^k`,
    `k = len(u)`).

    Derived as the exact inverse of the prover's basis fold
    `G^{(j+1)}_t = u_j⁻¹·G^{(j)}_t + u_j·G^{(j)}_{t+m}`: a coefficient `c` on a
    folded entry splits into `c·u_j⁻¹` on the low half and `c·u_j` on the high
    half, so unrolling from the collapsed scalar `[1]` back out gives
    `s ← concat(u_j⁻¹·s, u_j·s)` per round (rounds replayed last-to-first). Both
    the basis (`G`) and the evaluation vector (`b`) fold with this same low/high
    exponent pattern, so the one `s` serves both `⟨s, G⟩` and `⟨s, b⟩`."""
    k = u.shape[0]
    s = jnp.ones((1,), dtype=u.dtype)
    for j in range(k - 1, -1, -1):
        s = jnp.concatenate([u_inv[j] * s, u[j] * s])
    return s


def eval_challenge_poly(u: Array, u_inv: Array, x: Array) -> Array:
    """`g(x) = ∏_{j=0}^{k-1} (u_j⁻¹ + u_j · x^{2^{k-1-j}})` in O(k) — the folded
    scalar `b_final` without materializing the size-`n` `s` (the succinct read of
    `challenge_vector`). `x^{2^m}` comes from repeated squaring, so no field
    `pow` by a large exponent is needed."""
    k = u.shape[0]
    # squares[m] = x^{2^m}, m = 0 .. k-1
    squares = []
    cur = x
    for _ in range(k):
        squares.append(cur)
        cur = cur * cur
    acc = jnp.ones((), dtype=x.dtype)
    for j in range(k):
        acc = acc * (u_inv[j] + u[j] * squares[k - 1 - j])
    return acc
