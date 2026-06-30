# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Check-polynomial kernels behind the IPA fold — no EC point, no MSM.

The pieces a verifier (and an accumulation consumer) needs that touch *only* the
scalar field, factored out of `verifier.py` so they run and test on the scalar
field independent of the curve's `lax.msm` path (the same split KZG draws between
`_quotient_and_eval` and its MSMs). The generic monomial-basis vector
`b = (1, x, …, x^{n-1})` lives in `zorch.poly.univariate.powers`; the two kernels
here are IPA-specific:

- `challenge_vector` — the size-`n` vector `s` with `G_final = ⟨s, G⟩` (the one
  expensive MSM the verifier/decider owes). The dense coefficients of the check
  polynomial `h` below, built by the *exact* inverse of the prover's basis fold,
  so `⟨s, G⟩` reproduces the prover's collapsed basis by construction rather than
  by a re-derived closed form.
- `eval_challenge_poly` — `h(x) = ∏_j (1 + u_j · x^{2^{k-1-j}})`, the O(log n)
  evaluation of the check polynomial whose coefficients are `s`. This is the
  folded scalar `b` *without* materializing `s`, and the reason an accumulation
  step stays succinct: `h` is pinned by the `k = log n` challenges alone.

Both use the no-inverse form (`1 + u_j·X^…`, not `u_j⁻¹ + u_j·X^…`). That formula
IS the contract; it matches arkworks' check polynomial (the `poly-commit` crate's
`ipa_pc` succinct check, `compute_coeffs` / `evaluate`), pinned against that
oracle at zorch#339 W3 (see docs/pcs.md) so the decider's final-key MSM
byte-matches it — treat the arkworks symbol names as a pointer that may move, the
formula as the spec.

`challenge_vector` and `eval_challenge_poly` are two readings of the *same* object
— `eval_challenge_poly(u, x) == ⟨challenge_vector(u), powers(x, n)⟩` — and a test
pins that identity so the succinct path and the explicit path cannot drift.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def challenge_vector(u: Array) -> Array:
    """The size-`n` combiner `s` with `G_final = ⟨s, G⟩` and `b_final = ⟨s, b⟩`,
    where `u[j]` is round `j`'s challenge (`n = 2^k`, `k = len(u)`). These are the
    dense coefficients of the check polynomial `h`.

    Derived as the exact inverse of the prover's basis fold
    `G^{(j+1)}_t = G^{(j)}_t + u_j·G^{(j)}_{t+m}`: a coefficient `c` on a folded
    entry splits into `c` on the low half and `c·u_j` on the high half, so
    unrolling from the collapsed scalar `[1]` back out gives `s ← concat(s, u_j·s)`
    per round (rounds replayed last-to-first). Both the basis (`G`) and the
    evaluation vector (`b`) fold with this same low/high pattern, so the one `s`
    serves both `⟨s, G⟩` and `⟨s, b⟩`."""
    k = u.shape[0]
    s = jnp.ones((1,), dtype=u.dtype)
    for j in range(k - 1, -1, -1):
        s = jnp.concatenate([s, u[j] * s])
    return s


def eval_challenge_poly(u: Array, x: Array) -> Array:
    """`h(x) = ∏_{j=0}^{k-1} (1 + u_j · x^{2^{k-1-j}})` in O(k) — the folded scalar
    `b_final` without materializing the size-`n` `s` (the succinct read of
    `challenge_vector`). `x^{2^m}` comes from repeated squaring, so no field `pow`
    by a large exponent is needed."""
    k = u.shape[0]
    one = jnp.ones((), dtype=x.dtype)
    # squares[m] = x^{2^m}, m = 0 .. k-1
    squares = []
    cur = x
    for _ in range(k):
        squares.append(cur)
        cur = cur * cur
    acc = jnp.ones((), dtype=x.dtype)
    for j in range(k):
        acc = acc * (one + u[j] * squares[k - 1 - j])
    return acc
