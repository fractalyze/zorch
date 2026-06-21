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

from zorch import _composite
from zorch.fusion import fused_region
from zorch.hash.poseidon.linear import apply_dense_mds
from zorch.hash.poseidon.params import PoseidonParams

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
        return _permute_body(self, state, _composite.has_composite_op())


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
def _permute_body(perm: Poseidon, state: Array, has_composite_op: bool) -> Array:
    p = perm._p
    alpha = p.alpha
    w = perm.width
    half_full = p.full_rounds // 2
    partial = p.partial_rounds
    mds_rows = perm._mds_rows

    # +rc -> sbox(all lanes) -> MDS. ARC is elementwise add and the S-box is
    # jnp.power, so neither splits the kernel; the MDS rides as integer literals.
    def full_round(s: Array, rc: Array) -> Array:
        return apply_dense_mds(mds_rows, jnp.power(s + rc, alpha))

    # +rc -> sbox(last lane only) -> MDS.
    def partial_round(s: Array, rc: Array) -> Array:
        s = s + rc
        last = jnp.power(s[w - 1], alpha)
        # concatenate, not s.at[w-1].set: a static-index set lowers to scatter,
        # which would split the fused kernel.
        s = jnp.concatenate([s[: w - 1], last[None]])
        return apply_dense_mds(mds_rows, s)

    # The decomposition takes the Poseidon ABI operands explicitly so the marked
    # region carries them in order: state, then round constants flattened
    # row-major over all rounds.
    def permutation(
        s: Array,
        rc_flat: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata passed through on both the composite and
        # inline paths — the decomposition itself does not read it.
        total_rounds = 2 * half_full + partial
        rc = rc_flat.reshape(total_rounds, w)
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

    # ABI operands [state, round_constants flattened row-major].
    operands = (state, p.round_constants.reshape(-1))

    # The permutation shape rides as `composite.attributes` — the zkx
    # recognizer's contract: the four shape ints (it maps `alpha` to its s-box
    # degree) plus `mds`, the width*width MDS flattened row-major, which the
    # emitter applies as the dense linear layer (so the layer is params-driven,
    # not hardcoded). The body ignores them (metadata only).
    #
    # `mds` is a numpy int64 value (not a Python list/tuple) so it lowers to a
    # `dense<[..]> : tensor<N*Nxi64>` DenseElementsAttr the zkx recognizer parses
    # with GetCompositeAttrIntArray; a plain Python list/tuple lowers to an
    # unparsed ArrayAttr (`[..]`). Row-major width*width.
    marker_attrs: dict[str, object] = {
        "width": w,
        "full_rounds": p.full_rounds,
        "partial_rounds": partial,
        "alpha": alpha,
        "mds": np.array(
            [mds_rows[i][j] for i in range(w) for j in range(w)], dtype=np.int64
        ),
    }
    return fused_region(
        permutation,
        *operands,
        name=POSEIDON_MARKER,
        version=POSEIDON_MARKER_VERSION,
        **marker_attrs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Permutation] = Poseidon
