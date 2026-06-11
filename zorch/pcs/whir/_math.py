# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR-specific math shared by the prover and verifier.

Field-representation-safe by construction: every domain point comes from
`ReedSolomon.domain()` (actual field values), so the binary k-fold never needs the
subgroup generator as a host int — there is no Montgomery/canonical conversion to
get wrong. (Generic polynomial evaluation lives in `zorch.poly`; only the pieces
unique to WHIR's coset fold and per-round geometry live here.)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon, fri_fold_values
from zorch.poly.eq import expand_eq_to_hypercube


def round_code(code: ReedSolomon, fold: int) -> ReedSolomon:
    """The RS code whose message length is `code`'s shrunk by `2^fold` (same
    blowup, dtype, coset shift). WHIR round `r` queries `round_code(code,
    r·k_whir)`; the re-encode at the end of round `r` produces
    `round_code(code, (r+1)·k_whir)`."""
    blowup = code.block_len // code.message_len
    return ReedSolomon(
        message_len=code.message_len >> fold,
        blowup=blowup,
        dtype=code.dtype,
        coset_shift=code.coset_shift,
    )


def interp_quadratic_012(e0: Array, e1: Array, e2: Array, x: Array) -> Array:
    """Evaluate at `x` the degree-2 polynomial through (0,e0), (1,e1), (2,e2) — the
    sumcheck claim reduction (the round poly is a product of two multilinears).

    Pure field arithmetic rather than `poly.univariate.eval_univariate`: that
    helper's nested `compute_lagrange_basis` `@jit` + `jnp.dot` mis-lowers when
    inlined under the verifier's own `@jit` zone on this jax fork (eager-only).
    `eval_coeffs` (a single jitted power-chain) composes fine and is reused for the
    coefficient-form evaluations."""
    one = jnp.ones((), x.dtype)
    two = one + one
    s1 = e1 - e0
    s2 = e2 - e1
    p = (s2 - s1) / two
    q = s1 - p
    return (p * x + q) * x + e0


def pow2_powers(z: Array, n: int) -> list[Array]:
    """`[z, z², z⁴, …]` of length `n` — the powers-of-two point a multilinear's
    coefficient vector is evaluated on to realize a univariate evaluation at `z`."""
    out = [z]
    for _ in range(n - 1):
        out.append(out[-1] * out[-1])
    return out


def query_gamma_powers(gamma: Array, count: int) -> Array:
    """`[γ², γ³, …, γ^{count+1}]` — the per-query batching weights (the OOD term
    takes `γ¹`, so queries start at `γ²`). One stacked vector so a round's query
    contributions reduce in a single `(weights · terms).sum()`, not a Python
    accumulation loop. `count` (the query repetitions) is static."""
    out = [gamma * gamma]
    for _ in range(count - 1):
        out.append(out[-1] * gamma)
    return jnp.stack(out)


def eq_table(point: list[Array]) -> Array:
    """`eq(point, ·)` over the hypercube, indexed to match the weight table the
    sumcheck folds (LSB-first in `point`, mirroring the `[0::2]/[1::2]` fold
    order)."""
    one = jnp.ones((), point[0].dtype)
    return expand_eq_to_hypercube(jnp.stack(point[::-1]), one)


def binary_k_fold(values: Array, alphas: list[Array], coset_points: Array) -> Array:
    """Fold a query coset to the folded polynomial's value.

    `values` are the `2^k` opened codeword values on the coset
    `coset_points = {x·ω_k^j}` (conjugates a half apart: `coset_points[j+h] =
    −coset_points[j]`). `k` successive FRI 2-folds at `alphas` collapse them to the
    folded poly evaluated at `x^{2^k}`. Each step squares the surviving points,
    walking down the squared domains. Single-coset; `vmap` over the query axis."""
    vals = values
    pts = coset_points
    for alpha in alphas:
        h = vals.shape[0] // 2
        vals = fri_fold_values(vals[:h], vals[h:], alpha, pts[:h])
        pts = pts[:h] * pts[:h]
    return vals[0]
