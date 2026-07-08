# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Accumulator pre-computation for the small-value EqPoly sumcheck (Procedure 9).

Each round i ≤ l₀ of the small-value phase reads a precomputed table Aᵢ(v, u) that
already contracts every variable except the first i, so the round polynomial is a
cheap contraction Rᵢ·Aᵢ instead of a fresh pass over the cube. Aᵢ(v, u) sums
Πₖ pₖ(v, u, x) weighted by eq(w, ·) over the extended prefix v ∈ U_dⁱ⁻¹, node
u ∈ Û_d, and the bound suffix. precompute_accumulators fills all l₀ tables in one
sweep over β ∈ U_dˡ⁰; the index maps below route each β to the (round, v, u, y)
slots it contributes to.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, vmap

from zorch.sumcheck.domain import extend_to_round_domain
from zorch.utils.bits import log2_strict_usize


def _extend_prefix_to_domain(evals: Array, d: int, k: int) -> Array:
    """Extend the first k variables from {0,1} to the full U_d = {∞, 0, 1, …, d−1},
    leaving the rest on the cube: {p(β)} for β ∈ U_dᵏ × {0,1}ˡ⁻ᵏ, flattened. Each
    step lifts one variable, then reassembles lexicographic order by per-node
    slicing (not jnp.transpose, which the fork ignores on 3D arrays)."""
    l = log2_strict_usize(evals.shape[0])
    size = d + 1
    for i in range(k):
        prefix, suffix = size**i, 1 << (l - i - 1)
        reshaped = jnp.reshape(evals, (prefix, 2, suffix))
        lifted = extend_to_round_domain(
            reshaped[:, 0, :].reshape(-1), reshaped[:, 1, :].reshape(-1), d
        )
        parts = [jnp.reshape(lifted[n], (prefix, suffix)) for n in range(size)]
        evals = jnp.concatenate(parts, axis=1).reshape(-1)
    return evals


def precompute_accumulators(
    p_evals: Array, e_in: Array, e_out: list[Array]
) -> list[Array]:
    """Tables [A₁, …, Aₗ₀], Aᵢ of shape ((d+1)ⁱ⁻¹, d) over (v ∈ U_dⁱ⁻¹, u ∈ Û_d).
    e_in = eq(w_in, x_in) over {0,1}^{l/2}; e_out[i−1] = eq over (y, x_out) for
    round i (both the y and x_out domains, not x_out alone)."""
    d = p_evals.shape[0]
    l = log2_strict_usize(p_evals.shape[1])
    l_half = l // 2
    l_0 = len(e_out)
    x_in_size = 1 << l_half
    x_out_size = 1 << (l - l_half - l_0)
    beta_size = (d + 1) ** l_0

    # Extend the first l₀ variables to U_d, then contract x_in against eq(w_in, ·):
    # tA[β, x_out] = Σ_{x_in} e_in[x_in] · Πₖ pₖ(β, x_in, x_out).
    p_beta = vmap(lambda e: _extend_prefix_to_domain(e, d, l_0))(p_evals)
    p_beta_prod = jnp.reshape(
        jnp.prod(p_beta, axis=0), (beta_size, x_in_size, x_out_size)
    )
    t_a = (p_beta_prod * e_in[None, :, None]).sum(axis=1)  # (β, x_out)

    # _idx4 is pure host arithmetic, so collect each round's (β, flat v·d+u, y) slots
    # once on the host, then contract + scatter that round in a single vectorized op.
    # A per-contribution `.at[].add()` instead bakes O((d+1)^l₀·l₀) scatters into the
    # traced graph — the compile time/memory blow up exponentially in l₀.
    routes: list[tuple[list[int], list[int], list[int]]] = [
        ([], [], []) for _ in range(l_0)
    ]
    for beta_idx in range(beta_size):
        for i, v, u, y in _idx4(beta_idx, l_0, d):
            betas, vus, ys = routes[i - 1]
            betas.append(beta_idx)
            vus.append(v * d + u)
            ys.append(y)

    accumulators = []
    for i in range(l_0):
        betas, vus, ys = (jnp.asarray(col, dtype=jnp.int32) for col in routes[i])
        # contrib[k] = Σ_x_out e_out[i][y_k, :] · t_a[β_k, :]
        contrib = jnp.sum(e_out[i][ys, :] * t_a[betas, :], axis=1)
        flat = jnp.zeros((d + 1) ** i * d, dtype=p_evals.dtype).at[vus].add(contrib)
        accumulators.append(flat.reshape(((d + 1) ** i, d)))
    return accumulators


def _decode_beta(beta_idx: int, l_0: int, d: int) -> tuple[int, ...]:
    """β_idx → (β₁, …, βₗ₀), each in [0, d] (base-(d+1) digits, MSB first)."""
    digits = []
    for _ in range(l_0):
        digits.append(beta_idx % (d + 1))
        beta_idx //= d + 1
    return tuple(reversed(digits))


def _idx4(beta_idx: int, l_0: int, d: int) -> list[tuple[int, int, int, int]]:
    """Route β to its (i, v, u, y) accumulator slots (idx4, Definition A.5): split
    β = (v, u=βᵢ, y) at each round i, keeping only slots where u ∈ Û_d (u ≠ 1) and
    y stays on the boolean cube. U_d digit encoding: 0=∞, 1=0, 2=1, 3=2, ….
    Returns flat indices for v ∈ U_dⁱ⁻¹, u ∈ Û_d, y ∈ {0,1}ˡ⁰⁻ⁱ."""
    beta = _decode_beta(beta_idx, l_0, d)
    results = []
    for i in range(1, l_0 + 1):
        u_digit = beta[i - 1]
        if u_digit == 2:  # u = 1 is omitted from Û_d
            continue
        y_digits = beta[i:]
        if not all(y in (1, 2) for y in y_digits):  # y must be boolean (0 or 1)
            continue
        u = u_digit - 1 if u_digit > 2 else u_digit  # collapse the dropped u=1 slot
        y = sum((y_val - 1) << j for j, y_val in enumerate(y_digits))
        v = sum(v_val * ((d + 1) ** j) for j, v_val in enumerate(beta[: i - 1]))
        results.append((i, v, u, y))
    return results
