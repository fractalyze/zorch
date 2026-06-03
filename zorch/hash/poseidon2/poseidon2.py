"""Poseidon2 permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `jax.lax.composite`
(`fused_region`): zkx's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. With the standard external matrix the region is named
`poseidon2:W:E:I:S` and routes to zkx's dedicated, params-driven Poseidon2Fusion
emitter; a non-standard external matrix falls back to the generic
`zorch.fused_region` marker (whose generic LoopFusion would compile a full
permute into one register-spilling kernel). The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the linear layers use the
normal-form helpers (`apply_matrix`, `apply_internal`) so nothing lowers to a
reduce/dot/gather that would split the kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.fusion import FUSED_REGION_MARKER, fused_region
from zorch.hash.poseidon2.linear import (
    apply_external_standard,
    apply_internal,
    apply_matrix,
)
from zorch.hash.poseidon2.params import Poseidon2Params


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE fused region.
    """

    def __init__(self, params: Poseidon2Params) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Decided once here (eager): the matrix compare would stage into the
        # jaxpr if done inside the traced `permute` body. Gates both the marker
        # name and the const-free external layer.
        self._uses_standard_external = params.uses_standard_external_matrix
        self._fused_region_name = self._select_fused_region_name()

    def _select_fused_region_name(self) -> str:
        """Route to the dedicated Poseidon2Fusion only with the standard external
        matrix — its GPU emitter hardcodes the M4-circulant MDS, so a custom
        matrix would be silently ignored there; otherwise keep the generic marker
        (LoopFusion lowers the real body, staying correct).
        """
        p = self._p
        if self._uses_standard_external:
            return (
                f"poseidon2:{p.width}:{p.external_rounds}:{p.internal_rounds}:{p.alpha}"
            )
        return FUSED_REGION_MARKER

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got {state.shape}"
            )
        p = self._p
        alpha = p.alpha
        w, e_rounds, i_rounds = self.width, p.external_rounds, p.internal_rounds

        # The external MDS must not be a closed-over array on the named-emitter
        # path: jax.lax.composite lifts closed-over consts to leading operands, so
        # the matrix would leak in as a 7th operand and break the Poseidon2Fusion
        # 6-operand ABI. The standard matrix applies via integer literals (no
        # capture); a custom matrix takes the generic LoopFusion fallback, which
        # lowers the real body, so the closed array is harmless there.
        if self._uses_standard_external:

            def apply_external(s: Array) -> Array:
                return apply_external_standard(s)

        else:
            mds = p.external_matrix

            def apply_external(s: Array) -> Array:
                return apply_matrix(mds, s)

        # +rc -> sbox(all lanes) -> MDS
        def external_round(s: Array, rc: Array) -> Array:
            return apply_external(jnp.power(s + rc, alpha))

        # +rc(lane0) -> sbox(lane0) -> diffusion (off_diag scales the J term)
        def internal_round(s: Array, rc0: Array, diag: Array, off_diag: Array) -> Array:
            s0 = jnp.power(s[0] + rc0, alpha)
            # concatenate, not s.at[0].set: a static-index set lowers to scatter,
            # which would split the fused kernel.
            s = jnp.concatenate([s0[None], s[1:]])
            return apply_internal(diag, s, off_diag)

        # The decomposition takes the Poseidon2Fusion ABI operands explicitly so
        # the marked region carries them in order: round constants flattened
        # row-major, int_rc the lane-0 column, off_diag scaling the J term.
        def permutation(
            s: Array,
            ext_init_rc: Array,
            int_rc: Array,
            ext_term_rc: Array,
            diag: Array,
            off_diag: Array,
        ) -> Array:
            ext_init = ext_init_rc.reshape(e_rounds, w)
            ext_term = ext_term_rc.reshape(e_rounds, w)
            s = apply_external(s)  # initial pre-MDS
            for i in range(e_rounds):
                s = external_round(s, ext_init[i])
            for i in range(i_rounds):
                s = internal_round(s, int_rc[i], diag, off_diag)
            for i in range(e_rounds):
                s = external_round(s, ext_term[i])
            return s

        # ABI operands [state, ext_init_rc, int_rc, ext_term_rc, diag, off_diag].
        # zorch's internal matrix is J + Diag(internal_diag), so off_diag = 1.
        operands = (
            state,
            p.external_constants_initial.reshape(-1),
            p.internal_constants[:, 0],
            p.external_constants_terminal.reshape(-1),
            p.internal_diag,
            jnp.array(1, dtype=self.dtype),
        )

        return fused_region(permutation, *operands, name=self._fused_region_name)
