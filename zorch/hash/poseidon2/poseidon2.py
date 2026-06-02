"""Poseidon2 permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `jax.lax.composite`
named `zorch.round` (`fused_region`): zkx's `ZorchRoundRewriter` turns that
marker into a single custom-fusion kernel — one kernel by construction, not via
a per-hash compiler pattern match. The body is kept straight-line: rounds are
unrolled (fixed, small counts) and the linear layers use the normal-form helpers
(`apply_matrix`, `apply_internal`) so nothing lowers to a reduce/dot/gather that
would split the kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.fusion import fused_region
from zorch.hash.poseidon2.linear import apply_internal, apply_matrix
from zorch.hash.poseidon2.params import Poseidon2Params


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE fused region.
    """

    def __init__(self, params: Poseidon2Params):
        self._p = params
        self.width = params.width
        self.dtype = params.dtype

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got {state.shape}"
            )
        p = self._p
        mds, diag, alpha = p.external_matrix, p.internal_diag, p.alpha
        ext_init = p.external_constants_initial
        ext_term = p.external_constants_terminal
        int_rc = p.internal_constants

        def external_round(s, rc):  # +rc -> sbox(all lanes) -> MDS
            return apply_matrix(mds, jnp.power(s + rc, alpha))

        def internal_round(s, rc0):  # +rc(lane0) -> sbox(lane0) -> diffusion
            s0 = jnp.power(s[0] + rc0, alpha)
            # concatenate, not s.at[0].set: a static-index set lowers to scatter,
            # which would split the fused kernel.
            s = jnp.concatenate([s0[None], s[1:]])
            return apply_internal(diag, s)

        def permutation(s: Array) -> Array:
            s = apply_matrix(mds, s)  # initial pre-MDS
            for i in range(p.external_rounds):
                s = external_round(s, ext_init[i])
            for i in range(p.internal_rounds):
                s = internal_round(s, int_rc[i][0])
            for i in range(p.external_rounds):
                s = external_round(s, ext_term[i])
            return s

        return fused_region(permutation, state)
