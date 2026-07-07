"""Classic Poseidon permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `jax.lax.composite`
(`fused_region`): zkx's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. The region is named `zorch.poseidon` (distinct from
`zorch.poseidon2`), the permutation shape riding as `composite.attributes`
(`width`/`full_rounds`/`partial_rounds`/`alpha`/`mds`), and routes to zkx's
dedicated, params-driven Poseidon emitter. The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the dense MDS uses the normal-form
helper (`apply_dense_mds`) so nothing lowers to a reduce/dot/gather that would
split the kernel.

Classic Poseidon (ark-sponge style): each round is `ARC -> S-box -> dense MDS`.
The rounds split full/partial/full — `full_rounds/2` full rounds (S-box `x^alpha`
on all lanes), then `partial_rounds` partial rounds (S-box on the last lane
only), then `full_rounds/2` full rounds — and the dense MDS runs every round.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.fusion import fused_region
from zorch.hash.poseidon.linear import apply_dense_mds
from zorch.hash.poseidon.params import PoseidonParams
from zorch.hash.sponge import (
    _MODES,
    SPONGE_HASH_MARKER,
    SPONGE_HASH_MARKER_VERSION,
    SpongeType,
    _absorb,
)

if TYPE_CHECKING:
    from zorch.hash.permutation import Permutation

POSEIDON_MARKER = "zorch.poseidon"
# Marker revision riding as `composite.version`. zkx recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker.
POSEIDON_MARKER_VERSION = 1


class Poseidon:
    """A classic Poseidon permutation built from a PoseidonParams; implements
    Permutation.

    permute = full_rounds/2 full rounds (initial RC) -> partial_rounds partial
              rounds -> full_rounds/2 full rounds (terminal RC), each round
              `ARC -> S-box -> dense MDS`, as ONE fused region.
    """

    def __init__(self, params: PoseidonParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Extracted once here (eager): the MDS-to-canonical-int conversion would
        # stage into the jaxpr if done inside the traced `permute` body. The
        # const-free (literal-MDS) linear layer reads these ints; the marker
        # carries them (flattened row-major) as the `mds` attribute.
        self._mds_rows = params.mds_rows
        # Classic Poseidon always applies its dense MDS via integer literals and
        # routes to the dedicated `zorch.poseidon` emitter — there is no
        # free-form fallback (the MDS rides as a marker attribute either way).
        self.has_dedicated_fusion = True

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux
        # seat in `DuplexTranscript` (docs/conventions.md "Pytree registration").
        if not isinstance(other, Poseidon):
            return NotImplemented
        return self._p == other._p

    def __hash__(self) -> int:
        return hash(self._p)

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got {state.shape}"
            )
        return _permute_body(self, state)

    def sponge_hash(
        self,
        input: Array,
        rate: int,
        out: int,
        sponge_type: SpongeType = SpongeType.OVERWRITE,
    ) -> Array:
        """Fusion hook for `Sponge` — absorb+squeeze as ONE `zorch.sponge_hash`
        region the vendor expands into a register-resident kernel. `sponge_type`
        picks the construction (see `SpongeType`; `Sponge.hash` is the public
        API). Lowers under symbolic `len(input)` for export."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        return _sponge_hash_body(self, input, rate, out, sponge_type=sponge_type)


# The classic Poseidon permute on `s` given round constants flattened row-major
# (the marked region's ABI operand). Shared by the permute marker and the
# sponge-hash marker so the two carry one round schedule. full/partial/full
# rounds, each ARC -> S-box -> dense MDS; the MDS rides as integer literals.
def _permute_from_rc(perm: "Poseidon", s: Array, rc_flat: Array) -> Array:
    p = perm._p
    alpha = p.alpha
    w = perm.width
    half_full = p.full_rounds // 2
    partial = p.partial_rounds
    mds_rows = perm._mds_rows

    def full_round(st: Array, rc: Array) -> Array:
        return apply_dense_mds(mds_rows, jnp.power(st + rc, alpha))

    def partial_round(st: Array, rc: Array) -> Array:
        st = st + rc
        last = jnp.power(st[w - 1], alpha)
        # concatenate, not a static-index set: the latter lowers to scatter,
        # which would split the fused kernel.
        st = jnp.concatenate([st[: w - 1], last[None]])
        return apply_dense_mds(mds_rows, st)

    rc = rc_flat.reshape(2 * half_full + partial, w)
    r = 0
    for _ in range(half_full):
        s = full_round(s, rc[r])
        r += 1
    for _ in range(partial):
        s = partial_round(s, rc[r])
        r += 1
    for _ in range(half_full):
        s = full_round(s, rc[r])
        r += 1
    return s


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes (every
# Merkle level, leaf hash, and transcript observe/sample) — the uncached
# re-trace of this body dominated the first-trace-per-config floor (#216).
# The permutation is the static key, compared by value (#214); `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per permute) is unchanged.
@partial(jax.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Poseidon, state: Array) -> Array:
    p = perm._p
    w = perm.width

    def permutation(s: Array, rc_flat: Array, **_attrs: object) -> Array:
        # `_attrs` is marker metadata (name + attrs); the body ignores it.
        return _permute_from_rc(perm, s, rc_flat)

    # ABI operands [state, round_constants flattened row-major].
    operands = (state, p.round_constants.reshape(-1))
    # `mds` is a numpy int64 value (not a Python list) so it lowers to a
    # `dense<[..]>:tensor<N*Nxi64>` the recognizer parses with
    # GetCompositeAttrIntArray; a plain list lowers to an unparsed ArrayAttr.
    marker_attrs: dict[str, object] = _poseidon_marker_attrs(perm)
    return fused_region(
        permutation,
        *operands,
        name=POSEIDON_MARKER,
        version=POSEIDON_MARKER_VERSION,
        **marker_attrs,
    )


# The permute shape as `composite.attributes` (the recognizer's contract: shape
# ints — alpha is its s-box degree — plus `mds`, the width*width MDS flattened
# row-major, applied as the dense linear layer). Shared with the sponge marker.
def _poseidon_marker_attrs(perm: "Poseidon") -> dict[str, object]:
    p = perm._p
    w = perm.width
    return {
        "width": w,
        "full_rounds": p.full_rounds,
        "partial_rounds": p.partial_rounds,
        "alpha": p.alpha,
        "mds": np.array(perm._mds_rows, dtype=np.int64).flatten(),
    }


def _sponge_hash_body(
    perm: "Poseidon",
    input: Array,
    rate: int,
    out: int,
    sponge_type: SpongeType = SpongeType.OVERWRITE,
) -> Array:
    """Classic-Poseidon sponge: absorb+squeeze as ONE `zorch.sponge_hash` region.

    Byte-identical to `Sponge.hash`. The region carries the ABI operands
    [input, round_constants] explicitly (a `lax.composite` would lift the
    closed-over constants and break the operand ABI the recognizer expects). The
    `mds` rides as a marker attribute; `permutation="poseidon"` is the required
    discriminator that routes the recognizer to the classic-permute config arm.
    `sponge_type` selects the construction (same shared absorb + marker).
    """
    p = perm._p
    w = perm.width

    def sponge(inp: Array, rc_flat: Array, **_attrs: object) -> Array:
        state = jnp.zeros(w, dtype=inp.dtype)
        return _absorb(
            inp,
            state,
            rate,
            out,
            lambda s: _permute_from_rc(perm, s, rc_flat),
            sponge_type,
        )

    operands = (input, p.round_constants.reshape(-1))
    marker_attrs: dict[str, object] = {
        "permutation": "poseidon",
        "rate": rate,
        "digest_elems": out,
        **_poseidon_marker_attrs(perm),
    }
    marker_chained = _MODES[sponge_type].marker_chained
    if marker_chained:
        marker_attrs["chained"] = marker_chained
    return fused_region(
        sponge,
        *operands,
        name=SPONGE_HASH_MARKER,
        version=SPONGE_HASH_MARKER_VERSION,
        **marker_attrs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Permutation] = Poseidon
