"""Poseidon2 permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `jax.lax.composite`
(`fused_region`): zkx's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. With the standard external matrix the region is named
`zorch.poseidon2`, the permutation shape riding as `composite.attributes`
(`width`/`external_rounds`/`internal_rounds`/`alpha`), and routes to zkx's
dedicated, params-driven Poseidon2Fusion emitter; a non-standard external
matrix falls back to the generic
`zorch.fused_region` marker (whose generic LoopFusion would compile a full
permute into one register-spilling kernel). The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the linear layers use the
normal-form helpers (`apply_matrix`, `apply_internal`) so nothing lowers to a
reduce/dot/gather that would split the kernel.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch import _composite
from zorch.fusion import FUSED_REGION_MARKER, fused_region
from zorch.hash.poseidon2.linear import (
    apply_external_m4,
    apply_internal,
    apply_matrix,
)
from zorch.hash.poseidon2.params import Poseidon2Params

if TYPE_CHECKING:
    from zorch.hash.permutation import Permutation

POSEIDON2_MARKER = "zorch.poseidon2"
# Marker revision riding as `composite.version`. zkx recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker.
POSEIDON2_MARKER_VERSION = 1


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE fused region.
    """

    def __init__(self, params: Poseidon2Params) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Decided once here (eager): the structure check + M4 extraction would
        # stage into the jaxpr if done inside the traced `permute` body. Gates both
        # the marker name and the const-free (literal-M4) external layer.
        self._is_m4_structured = params.is_m4_block_structured
        self._external_m4 = params.external_m4 if self._is_m4_structured else None
        self._fused_region_name = self._select_fused_region_name()
        # Dedicated == permute lowers to a hash-named marker, not the generic
        # region one (which a vendor can't route, so a whole-region composite
        # around it is unexpandable). Derived from the marker choice itself so the
        # two can't drift if `_select_fused_region_name` grows another case.
        self.has_dedicated_fusion = self._fused_region_name != FUSED_REGION_MARKER

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux
        # seat in `DuplexTranscript` (docs/conventions.md "Pytree registration").
        if not isinstance(other, Poseidon2):
            return NotImplemented
        return self._p == other._p

    def __hash__(self) -> int:
        return hash(self._p)

    def _select_fused_region_name(self) -> str:
        """Route to the dedicated Poseidon2Fusion for any `(I + J_blocks) ⊗ M4`
        external matrix — the emitter applies the M4 the marker carries as an
        attribute, so Plonky3's `circ(2,3,1,1)` and the HorizenLabs matrix both
        take the dedicated path. A free-form matrix is not M4-block-structured, so
        it keeps the generic marker (LoopFusion lowers the real body, staying
        correct, just slow to compile).
        """
        if self._is_m4_structured:
            return POSEIDON2_MARKER
        return FUSED_REGION_MARKER

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got {state.shape}"
            )
        return _permute_body(self, state, _composite.has_composite_op())

    def permute_batched(self, states: Array) -> Array:
        """Permute a batch of states `(n, width) -> (n, width)`, numerically a
        `vmap(permute)` over the leading axis.

        On the dedicated marker this emits ONE `zorch.poseidon2` region over the
        whole `(n, width)` batch whose decomposition is `lax.map` over a single
        SHARED `(width,)` permute body, instead of `vmap` baking the batch into a
        fresh per-`n` decomposition. The region's operands (batched state plus the
        scalar/vector round constants) are byte-for-byte what `vmap(permute)`
        emits, so a marker-routed vendor emits the same batched kernel — the only
        change is that the ragged Merkle-fold rounds share one lowered permute
        body rather than re-emitting it per round. A free-form
        (non-M4) matrix keeps the generic `vmap` path: its body must stay
        inline-straight-line for the single-kernel fallback, which a `lax.map`
        call would break."""
        if states.ndim != 2 or states.shape[1] != self.width:
            raise ValueError(
                f"states must be 2-D of shape (n, {self.width}), got {states.shape}"
            )
        if not self.has_dedicated_fusion:
            return jax.vmap(self.permute)(states)
        return _permute_body_batched(self, states, _composite.has_composite_op())


def _permutation_body(
    perm: Poseidon2,
    s: Array,
    ext_init_rc: Array,
    int_rc: Array,
    ext_term_rc: Array,
    diag: Array,
    off_diag: Array,
) -> Array:
    """The straight-line permute on a single `(width,)` state, taking the
    Poseidon2Fusion ABI operands explicitly: round constants flattened row-major,
    int_rc the lane-0 column, off_diag scaling the internal J term.

    The decomposition every `zorch.poseidon2` region runs — spliced inline on the
    single-state path, or referenced as a shared callable (`_single_permute`) by
    the batched path so the ragged Merkle-fold rounds share one lowered copy."""
    p = perm._p
    alpha = p.alpha
    w, e_rounds, i_rounds = perm.width, p.external_rounds, p.internal_rounds

    # The external MDS must not be a closed-over array on the named-emitter
    # path: jax.lax.composite lifts closed-over consts to leading operands, so
    # the matrix would leak in as a 7th operand and break the Poseidon2Fusion
    # 6-operand ABI. An M4-block-structured matrix applies via integer literals
    # (the 4×4 M4, no array capture) and rides as a marker attribute; a free-form
    # matrix takes the generic LoopFusion fallback, which lowers the real body, so
    # the closed array is harmless there.
    if perm._is_m4_structured:
        m4 = perm._external_m4
        assert m4 is not None  # _is_m4_structured ⇒ M4 was extracted

        def apply_external(state: Array) -> Array:
            return apply_external_m4(state, m4)

    else:
        mds = p.external_matrix

        def apply_external(state: Array) -> Array:
            return apply_matrix(mds, state)

    # +rc -> sbox(all lanes) -> MDS
    def external_round(state: Array, rc: Array) -> Array:
        return apply_external(jnp.power(state + rc, alpha))

    # +rc(lane0) -> sbox(lane0) -> diffusion (off_diag scales the J term)
    def internal_round(state: Array, rc0: Array) -> Array:
        s0 = jnp.power(state[0] + rc0, alpha)
        # concatenate, not state.at[0].set: a static-index set lowers to scatter,
        # which would split the fused kernel.
        state = jnp.concatenate([s0[None], state[1:]])
        return apply_internal(diag, state, off_diag)

    ext_init = ext_init_rc.reshape(e_rounds, w)
    ext_term = ext_term_rc.reshape(e_rounds, w)
    s = apply_external(s)  # initial pre-MDS
    for i in range(e_rounds):
        s = external_round(s, ext_init[i])
    for i in range(i_rounds):
        s = internal_round(s, int_rc[i])
    for i in range(e_rounds):
        s = external_round(s, ext_term[i])
    return s


# Shared, NON-inlined single-state permute: the batched decomposition `lax.map`s
# this one callable, and JAX lowers an identical-aval subfunction to ONE
# `func.func` (referenced by every batched composite) rather than re-emitting the
# ~width-sized body per ragged fold round. `inline=True` would splice it back into
# each `lax.map` body and defeat the sharing. `perm` is the static value key (#214).
@partial(jax.jit, static_argnames=("perm",), inline=False)
def _single_permute(
    perm: Poseidon2,
    s: Array,
    ext_init_rc: Array,
    int_rc: Array,
    ext_term_rc: Array,
    diag: Array,
    off_diag: Array,
) -> Array:
    return _permutation_body(perm, s, ext_init_rc, int_rc, ext_term_rc, diag, off_diag)


def _abi_operands(perm: Poseidon2, state: Array) -> tuple[Array, ...]:
    """The Poseidon2Fusion ABI operands [state, ext_init_rc, int_rc, ext_term_rc,
    diag, off_diag]; `state` is `(width,)` single or `(n, width)` batched, the
    constants identical either way. The internal matrix is
    internal_j_scale*J + Diag(internal_diag); the ABI's off_diag operand carries
    the J scale (params normalize None to 1)."""
    p = perm._p
    return (
        state,
        p.external_constants_initial.reshape(-1),
        p.internal_constants[:, 0],
        p.external_constants_terminal.reshape(-1),
        p.internal_diag,
        p.internal_j_scale,
    )


def _marker_attrs(perm: Poseidon2) -> tuple[dict[str, object], int]:
    """The dedicated marker's `composite.attributes` + version. On the dedicated
    marker the permutation shape rides as attributes — the zkx recognizer's
    contract: the four shape ints (it maps `alpha` to its s-box degree) plus
    `external_m4`, the 4×4 base M4 flattened row-major, which the emitter applies
    per 4-block (so the external layer is no longer hardcoded). The body ignores
    them (metadata only); the generic marker stays attrs-free."""
    if not perm.has_dedicated_fusion:
        return {}, 0
    assert perm._external_m4 is not None  # has_dedicated_fusion ⇒ M4-structured
    attrs: dict[str, object] = {
        "width": perm.width,
        "external_rounds": perm._p.external_rounds,
        "internal_rounds": perm._p.internal_rounds,
        "alpha": perm._p.alpha,
        # A numpy value (not a Python list) so it lowers to a
        # `dense<[..]> : tensor<16xi64>` attribute the zkx recognizer parses with
        # GetCompositeAttrIntArray (a plain list lowers to an unparsed ArrayAttr).
        # Row-major 4×4.
        "external_m4": np.array(
            [perm._external_m4[r][c] for r in range(4) for c in range(4)],
            dtype=np.int64,
        ),
    }
    return attrs, POSEIDON2_MARKER_VERSION


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes (every
# Merkle level, leaf hash, and transcript observe/sample) — the uncached
# re-trace of this body dominated the first-trace-per-config floor (#216).
# The permutation is the static key, compared by value (#214); `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per permute) is unchanged. `has_composite_op` is a
# pure cache key: `composite_or_inline` reads the flag itself at trace time,
# but the traced body differs across its values (marker vs inlined fallback),
# so a flip must not replay a stale entry.
@partial(jax.jit, static_argnames=("perm", "has_composite_op"), inline=True)
def _permute_body(perm: Poseidon2, state: Array, has_composite_op: bool) -> Array:
    def decomposition(
        s: Array,
        ext_init_rc: Array,
        int_rc: Array,
        ext_term_rc: Array,
        diag: Array,
        off_diag: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata passed through on both the composite and
        # inline paths — the decomposition itself does not read it. Inlined here
        # so the single-state region stays one straight-line body (the generic
        # marker's single-kernel requirement allows no call).
        return _permutation_body(
            perm, s, ext_init_rc, int_rc, ext_term_rc, diag, off_diag
        )

    attrs, version = _marker_attrs(perm)
    return fused_region(
        decomposition,
        *_abi_operands(perm, state),
        name=perm._fused_region_name,
        version=version,
        **attrs,
    )


# Batched twin of `_permute_body`: one region over the whole `(n, width)` batch
# whose decomposition `lax.map`s the SHARED `_single_permute`. The region's
# operands match `vmap(permute)` byte-for-byte (batched state + the same
# unbatched constants), so a marker-routed vendor emits the same batched kernel;
# the win is purely that every ragged fold round references one lowered body.
# Dedicated marker only (see `permute_batched` for why the generic path vmaps).
@partial(jax.jit, static_argnames=("perm", "has_composite_op"), inline=True)
def _permute_body_batched(
    perm: Poseidon2, states: Array, has_composite_op: bool
) -> Array:
    def decomposition(
        batched: Array,
        ext_init_rc: Array,
        int_rc: Array,
        ext_term_rc: Array,
        diag: Array,
        off_diag: Array,
        **_attrs: object,
    ) -> Array:
        return jax.lax.map(
            lambda s: _single_permute(
                perm, s, ext_init_rc, int_rc, ext_term_rc, diag, off_diag
            ),
            batched,
        )

    attrs, version = _marker_attrs(perm)
    return fused_region(
        decomposition,
        *_abi_operands(perm, states),
        name=perm._fused_region_name,
        version=version,
        **attrs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Permutation] = Poseidon2
