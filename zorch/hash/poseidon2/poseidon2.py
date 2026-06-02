"""Poseidon2 permutation — scheme-agnostic, normal-form, one function.

Linear layers are static-Python-sliced add/double trees: they emit no
`dot`/`reduce`/`gather`, so the permutation lowers as fusion-friendly
element-wise ops. The permutation class (later task) applies them via
`lax.fori_loop` over isolated round bodies.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax

from zorch.hash.poseidon2.params import Poseidon2Params, _mds_external_default


def _pow(x, alpha: int):
    """S-box x^alpha by repeated multiply (alpha small + static; fusion-friendly)."""
    result = x
    for _ in range(alpha - 1):
        result = result * x
    return result


def _external_linear(x: Array, w: int) -> Array:
    """Canonical Poseidon2 external MDS (M4-circulant, 2x diagonal block).

    out[4c+r] = (M4 @ chunk_c)[r] + sum_d (M4 @ chunk_d)[r]. Pure
    adds/doublings, static slices -> no dot/reduce/gather.
    """
    nc = w // 4

    def m4(c0, c1, c2, c3):
        s = (c0 + c1) + (c2 + c3)  # 4-sum as a chained add-tree
        return (
            s + c0 + (c1 + c1),  # row r = s + c[r] + 2*c[r+1 mod 4]
            s + c1 + (c2 + c2),
            s + c2 + (c3 + c3),
            s + c3 + (c0 + c0),
        )

    y = [m4(x[4 * b], x[4 * b + 1], x[4 * b + 2], x[4 * b + 3]) for b in range(nc)]
    out = [None] * w
    for r in range(4):
        s = y[0][r]
        for b in range(1, nc):
            s = s + y[b][r]  # per-row chunk-sum, chained adds
        for c in range(nc):
            out[4 * c + r] = y[c][r] + s
    return jnp.stack(out)


def _internal_linear(x: Array, diag_m1: Array, w: int) -> Array:
    """Internal diffusion J + Diag(V) in normal form: out[i] = full_sum + V[i]*x[i]."""
    full_sum = x[0]
    for j in range(1, w):
        full_sum = full_sum + x[j]  # chained add-tree, not jnp.sum
    return jnp.stack([full_sum + diag_m1[i] * x[i] for i in range(w)])


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE function. Rounds run via
              lax.fori_loop over isolated bodies — the exact shape a future
              fused_rounds(body, n) will wrap.
    """

    def __init__(self, params: Poseidon2Params):
        # The normal-form external path is hardcoded for the canonical M4-circulant;
        # a custom external_matrix override is deferred.
        canonical = _mds_external_default(params.width, params.dtype)
        if not bool(jnp.array_equal(params.external_matrix, canonical)):
            raise NotImplementedError(
                "custom external_matrix override not supported in the normal-form path"
            )
        self._p = params
        self.width = params.width
        self.dtype = params.dtype

    def permute(self, state: Array) -> Array:
        p = self._p
        w, alpha, diag = p.width, p.alpha, p.internal_diag

        def external_body(rc):  # full round: +rc -> sbox(all) -> MDS
            def body(i, s):
                s = s + rc[i]
                s = _pow(s, alpha)
                return _external_linear(s, w)

            return body

        def internal_body(
            i, s
        ):  # partial round: +rc(lane0) -> sbox(lane0) -> diffusion
            s = s + p.internal_constants[i]
            s0 = _pow(s[0], alpha)
            s = jnp.concatenate([s0[None], s[1:]])
            return _internal_linear(s, diag, w)

        state = _external_linear(state, w)  # initial pre-MDS
        state = lax.fori_loop(
            0, p.external_rounds, external_body(p.external_constants_initial), state
        )
        state = lax.fori_loop(0, p.internal_rounds, internal_body, state)
        state = lax.fori_loop(
            0, p.external_rounds, external_body(p.external_constants_terminal), state
        )
        return state
