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

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

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

# Whole-sponge marker: a padding-free overwrite absorb + squeeze as ONE region the
# vendor expands into the fused `poseidon2_sponge_hash` kernel — state stays
# register-resident across every absorb block, instead of a per-block permute
# chain that round-trips the state through DRAM (dominates a wide-column leaf
# hash). Same 6-operand Poseidon2Fusion ABI as the permute marker; the absorb
# shape rides as attributes and the kernel reads the absorb length at runtime, so
# one cubin serves every leaf width AND a symbolic width exports (no Python branch
# on the block count).
SPONGE_HASH_MARKER = "zorch.poseidon2_sponge_hash"
SPONGE_HASH_MARKER_VERSION = 1


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
        """Absorb `input` (1-D) in `rate`-blocks and squeeze `out` lanes as ONE
        `zorch.poseidon2_sponge_hash` region the vendor expands into the fused
        sponge kernel (state register-resident across blocks) instead of a
        per-block permute chain. Only the dedicated (M4-structured) path emits the
        marker; otherwise the inline absorb runs (a generic permutation has no
        fused sponge kernel). Byte-identical to `Sponge.hash`'s padding-free
        overwrite absorb, and lowers under a symbolic `len(input)` — the kernel
        reads the absorb length at runtime — for recompile-free export."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        return _sponge_hash_body(self, input, rate, out)


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


def _absorb_symbolic(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
) -> Array:
    """Shape-polymorphic padding-free absorb: one `while_loop` over `ceil(n/rate)`
    blocks, no Python branch on `n`. Each block overwrites its `w = min(rate, n -
    i*rate)` real lanes then permutes — byte-identical to the unrolled absorb. The
    sponge-hash marker's decomposition for a symbolic `n` (export); the vendor
    kernel does the real runtime loop."""
    n = input.shape[0]
    nb = (n + rate - 1) // rate  # ceil(n / rate)
    lanes = jnp.arange(rate)

    def cond(carry: tuple[Array, Array]) -> Array:
        return carry[1] < nb

    def body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        s, i = carry
        start = i * rate
        w = jnp.minimum(rate, n - start)
        # The last block reads past n; clamp OOB indices in-bounds (those lanes
        # are masked out below, so the clamped values are discarded).
        block = input[jnp.clip(start + lanes, 0, n - 1)]
        s = s.at[:rate].set(jnp.where(lanes < w, block, s[:rate]))
        return permute(s), i + 1

    state, _ = lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state[:out]


def _sponge_hash_body(perm: Poseidon2, input: Array, rate: int, out: int) -> Array:
    """Emit the `zorch.poseidon2_sponge_hash` marker (dedicated path) or run the
    inline absorb (generic path). The decomposition is byte-identical to
    `Sponge.hash` — overwrite the leading lanes with each block then permute, a
    partial last block overwriting only its own lanes — so the marked region's
    fallback HLO matches the kernel. The 6-operand region carries the round
    constants explicitly (a `lax.composite` would otherwise lift closed-over
    consts as extra operands and break the Poseidon2Fusion ABI)."""
    w = perm.width
    absorb_len = input.shape[0]
    symbolic = not isinstance(absorb_len, int)

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

        state = jnp.zeros(w, dtype=inp.dtype)
        if symbolic:  # shape-poly export: while_loop over the blocks
            return _absorb_symbolic(inp, state, rate, out, permute)
        n = absorb_len
        if n == 0:
            return state[:out]
        if n <= rate:  # single (possibly partial) block
            state = state.at[:n].set(inp[:n])
            return permute(state)[:out]
        state = state.at[:rate].set(inp[:rate])
        state = permute(state)
        full = n // rate
        for i in range(1, full):  # remaining full blocks (unrolled; n is static)
            state = state.at[:rate].set(inp[i * rate : (i + 1) * rate])
            state = permute(state)
        tail = n - full * rate
        if tail:  # partial last block overwrites only its own lanes
            state = state.at[:tail].set(inp[full * rate :])
            state = permute(state)
        return state[:out]

    operands = _abi_operands(perm, input)
    # A generic permutation has no fused sponge kernel — run the absorb inline (a
    # LoopFusion over a whole sponge would register-spill); only the dedicated
    # path emits the marker. The absorb shape rides as attributes; the body
    # ignores them (the kernel reads width/rate/rounds from the backend config and
    # the absorb length from the operand shape at runtime).
    if not perm.has_dedicated_fusion:
        return sponge(*operands)
    p = perm._p
    marker_attrs: dict[str, int] = {
        "width": w,
        "rate": rate,
        "digest_elems": out,
        "external_rounds": p.external_rounds,
        "internal_rounds": p.internal_rounds,
        "alpha": p.alpha,
    }
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
