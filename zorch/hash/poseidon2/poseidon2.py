"""Poseidon2 permutation — scheme-agnostic, written as readable matmul + power.

The permutation is one function (all rounds) in its natural algebraic form:
`external_matrix @ state` for the full rounds, `internal_matrix @ state` for the
partial rounds, `state ** alpha` for the S-box. This is reference semantics; the
zkx compiler recognizes the pattern and lowers the whole permutation to a single
fused kernel.

Marking `permute` explicitly with a `stablehlo.composite` is the by-construction
alternative to that pattern match, but the marker op is not yet wired through the
zkx/StableHLO-fork lowering.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax

from zorch.hash.poseidon2.params import Poseidon2Params


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE function. Rounds run via
              lax.fori_loop over isolated bodies.
    """

    def __init__(self, params: Poseidon2Params):
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Internal diffusion matrix M_I = J + Diag(internal_diag); internal_diag
        # (the V vector) is the internal layer's only free part, so the matrix is
        # derived once here rather than carried on params.
        w = params.width
        self._internal_matrix = jnp.ones((w, w), dtype=params.dtype) + jnp.diag(
            params.internal_diag
        )

    def permute(self, state: Array) -> Array:
        p = self._p
        mds, m_int, alpha = p.external_matrix, self._internal_matrix, p.alpha

        def external_body(rc):  # full round: +rc -> sbox(all lanes) -> MDS
            def body(i, s):
                return mds @ jnp.power(s + rc[i], alpha)

            return body

        def internal_body(
            i, s
        ):  # partial round: +rc(lane0) -> sbox(lane0) -> diffusion
            s = s + p.internal_constants[i]
            s = s.at[0].set(jnp.power(s[0], alpha))
            return m_int @ s

        state = mds @ state  # initial pre-MDS
        state = lax.fori_loop(
            0, p.external_rounds, external_body(p.external_constants_initial), state
        )
        state = lax.fori_loop(0, p.internal_rounds, internal_body, state)
        state = lax.fori_loop(
            0, p.external_rounds, external_body(p.external_constants_terminal), state
        )
        return state
