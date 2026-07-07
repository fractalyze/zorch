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

from zorch.fusion import FUSED_REGION_MARKER, fused_region
from zorch.hash.poseidon2.linear import (
    apply_external_m4,
    apply_internal,
    apply_matrix,
)
from zorch.hash.poseidon2.params import Poseidon2Params
from zorch.hash.sponge import (
    SPONGE_HASH_MARKER,
    SPONGE_HASH_MARKER_VERSION,
    _absorb_symbolic,
)

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
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must match the permutation field "
                f"{self.dtype}"
            )
        return _permute_body(self, state)

    def sponge_hash(self, input: Array, rate: int, out: int) -> Array:
        """Absorb `input` and squeeze `out` lanes as ONE `sponge_hash`
        region (fused kernel, state register-resident) — byte-identical to
        `Sponge.hash`. Only the dedicated (M4-structured) path emits the marker; a
        generic permutation runs the inline absorb. Lowers under symbolic
        `len(input)` (the kernel reads the length at runtime) for export."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        return _sponge_hash_body(self, input, rate, out)

    def linear_hash(self, input: Array, rate: int, out: int) -> Array:
        """Chained (Merkle-Damgard) hash — absorb in `rate`-sized blocks,
        zero-padding a short final block, and carry the prior block's digest
        (`state[:out]`) in the capacity lanes before each permute. The digest
        fills the whole capacity, so `rate + out == width`. Concrete
        `len(input)` only (no symbolic export)."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        if rate + out != self.width:
            raise ValueError(
                f"chained hash needs rate + out == width (the digest fills the "
                f"capacity), got {rate} + {out} != {self.width}"
            )
        return _sponge_hash_body(self, input, rate, out, chained=True)


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

    The decomposition every `zorch.poseidon2` region runs, spliced inline (the
    generic marker's single-kernel requirement allows no call). A batch is
    `vmap(permute)`, which lowers to the same marker over a batched operand."""
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


def _external_m4_attr(perm: Poseidon2) -> np.ndarray:
    """The base M4 flattened row-major as a numpy int64 `(16,)`. A numpy value
    (not a Python list) so it lowers to a `dense<[..]> : tensor<16xi64>` the zkx
    recognizer can parse (a plain list lowers to an unparsed ArrayAttr). Caller
    guards on `has_dedicated_fusion`."""
    assert perm._external_m4 is not None  # has_dedicated_fusion ⇒ M4-structured
    return np.array(perm._external_m4, dtype=np.int64).flatten()


def _marker_attrs(perm: Poseidon2) -> tuple[dict[str, object], int]:
    """The dedicated marker's `composite.attributes` + version. The permutation
    shape rides as attributes — the zkx recognizer's contract: the four shape ints
    (it maps `alpha` to its s-box degree) plus `external_m4`, the 4×4 base M4. The
    recognizer rewrites only the M4 its kernel implements (the canonical
    `circ(2,3,1,1)`) and leaves any other matrix to inline its real body, so the
    attr is what identifies the matrix. The body ignores these attrs (metadata
    only); the generic marker stays attrs-free."""
    if not perm.has_dedicated_fusion:
        return {}, 0
    attrs: dict[str, object] = {
        "width": perm.width,
        "external_rounds": perm._p.external_rounds,
        "internal_rounds": perm._p.internal_rounds,
        "alpha": perm._p.alpha,
        "external_m4": _external_m4_attr(perm),
    }
    return attrs, POSEIDON2_MARKER_VERSION


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes (every
# Merkle level, leaf hash, and transcript observe/sample) — the uncached
# re-trace of this body dominated the first-trace-per-config floor (#216).
# The permutation is the static key, compared by value (#214); `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per permute) is unchanged.
@partial(jax.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Poseidon2, state: Array) -> Array:
    def decomposition(
        s: Array,
        ext_init_rc: Array,
        int_rc: Array,
        ext_term_rc: Array,
        diag: Array,
        off_diag: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata passed through — the decomposition itself
        # does not read it. Inlined here so the single-state region stays one
        # straight-line body (the generic marker's single-kernel requirement
        # allows no call).
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


def _sponge_hash_body(
    perm: Poseidon2, input: Array, rate: int, out: int, chained: bool = False
) -> Array:
    """Emit the `sponge_hash` marker (dedicated path) or run the
    `while_loop` absorb (generic). The decomposition is byte-identical to
    `Sponge.hash` so the region's fallback HLO matches the kernel. The 6-operand
    region carries
    the round constants explicitly — a `lax.composite` would lift closed-over
    consts as extra operands and break the Poseidon2Fusion ABI."""
    w = perm.width

    def sponge(
        inp: Array,
        ext_init_rc: Array,
        int_rc: Array,
        ext_term_rc: Array,
        diag: Array,
        off_diag: Array,
        **_attrs: object,
    ) -> Array:
        def permute(s: Array) -> Array:
            return _permutation_body(
                perm, s, ext_init_rc, int_rc, ext_term_rc, diag, off_diag
            )

        # One `while_loop` absorb for any length, concrete or symbolic. A static
        # unroll traced once per emission (`lax.composite` has no trace cache), so
        # a wide leaf paid O(blocks) trace + compile for a body the recognizer
        # discards anyway — it matches the marker by name + attrs, not its shape.
        state = jnp.zeros(w, dtype=inp.dtype)
        if chained:
            # Chained (Merkle-Damgard) hash: each block is [data(rate,
            # zero-padded) | prior-digest(out)] (rate + out == width), permuted.
            # Block 0's zero capacity falls out of the zeroed initial state.
            # Concrete n only.
            #
            # Written const-free (only the one `jnp.zeros(w)` state init, like the
            # `_absorb_symbolic` path below): jax.lax.composite lifts every traced
            # const to a leading operand, which would break the Poseidon2 6-operand
            # ABI. So the block lanes come from `.at[].set(inp slice)`, the capacity
            # from a slice, and the partial-tail zero-pad from field
            # self-subtraction (`pad - pad`) rather than a fresh `jnp.zeros`.
            n = input.shape[0]
            if not isinstance(n, int):  # shape-poly export unsupported for chained
                raise NotImplementedError(
                    "chained linear hash does not support symbolic-n export"
                )
            if n == 0:
                return state[:out]
            num_blocks = (n + rate - 1) // rate  # permute every block
            for blk in range(num_blocks):
                start = blk * rate
                count = min(rate, n - start)
                cap = state[:out]  # prior digest (zeros on block 0)
                state = state.at[:count].set(inp[start : start + count])
                if count < rate:  # zero-pad the partial tail (no const)
                    pad = state[count:rate]
                    state = state.at[count:rate].set(pad - pad)
                state = state.at[rate : rate + out].set(cap)  # chain
                state = permute(state)
            return state[:out]
        return _absorb_symbolic(inp, state, rate, out, permute)

    operands = _abi_operands(perm, input)
    # Generic permutation: no fused sponge kernel, run the absorb directly (a
    # whole-sponge LoopFusion register-spills); only the dedicated path marks it.
    if not perm.has_dedicated_fusion:
        return sponge(*operands)
    p = perm._p
    # external_m4 rides here too (like the permute marker) so the recognizer can
    # gate the sponge kernel on the M4 it implements — the canonical circ(2,3,1,1)
    # rewrites, any other M4 inlines its real body.
    marker_attrs: dict[str, object] = {
        "permutation": "poseidon2",
        "width": w,
        "rate": rate,
        "digest_elems": out,
        "external_rounds": p.external_rounds,
        "internal_rounds": p.internal_rounds,
        "alpha": p.alpha,
        "external_m4": _external_m4_attr(perm),
    }
    if chained:
        # Chained selects EmitLinearHashAbsorb; encoded as an int (1) since bool
        # composite attributes have no precedent. external_m4 already rides in
        # marker_attrs above (via _external_m4_attr), matching the
        # decomposition's apply_external_m4.
        marker_attrs["chained"] = 1
    return fused_region(
        sponge,
        *operands,
        name=SPONGE_HASH_MARKER,
        version=SPONGE_HASH_MARKER_VERSION,
        **marker_attrs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Permutation] = Poseidon2
