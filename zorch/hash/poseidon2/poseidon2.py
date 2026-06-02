"""Poseidon2 permutation — scheme-agnostic, normal-form, one function.

Linear layers are static-Python-sliced add/double trees: they emit no
`dot`/`reduce`/`gather`, so the permutation lowers as fusion-friendly
element-wise ops. The permutation class (later task) applies them via
`lax.fori_loop` over isolated round bodies.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _external_linear(x: Array, w: int) -> Array:
    """Canonical Poseidon2 external MDS in normal form (M4-circulant, 2x diagonal block).

    out[4c+r] = (M4 @ chunk_c)[r] + sum_d (M4 @ chunk_d)[r]. Pure adds/doublings,
    static slices -> no dot/reduce/gather.
    """
    nc = w // 4

    def m4(c0, c1, c2, c3):
        s = (c0 + c1) + (c2 + c3)                 # 4-sum as a chained add-tree
        return (s + c0 + (c1 + c1),               # row r = s + c[r] + 2*c[r+1 mod 4]
                s + c1 + (c2 + c2),
                s + c2 + (c3 + c3),
                s + c3 + (c0 + c0))

    y = [m4(x[4 * b], x[4 * b + 1], x[4 * b + 2], x[4 * b + 3]) for b in range(nc)]
    out = [None] * w
    for r in range(4):
        s = y[0][r]
        for b in range(1, nc):
            s = s + y[b][r]                        # per-row chunk-sum, chained adds
        for c in range(nc):
            out[4 * c + r] = y[c][r] + s
    return jnp.stack(out)


def _internal_linear(x: Array, diag_m1: Array, w: int) -> Array:
    """Internal diffusion J + Diag(V) in normal form: out[i] = full_sum + V[i]*x[i]."""
    full_sum = x[0]
    for j in range(1, w):
        full_sum = full_sum + x[j]                 # chained add-tree, not jnp.sum
    return jnp.stack([full_sum + diag_m1[i] * x[i] for i in range(w)])
