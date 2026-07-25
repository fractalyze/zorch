"""Optimized-sparse Poseidon permutation — a naive Hades Poseidon, faster.

`SparsePoseidon` re-factors the partial rounds of a naive Hades Poseidon: a
transition matrix `P` after the last pre-partial round, then a rank-structured
sparse update each partial round, instead of the dense MDS. It produces the same
permutation that *naive* schedule would with the equivalent (unfolded) constants
— dense MDS every round, full-width constants — but with fewer multiplications,
so it consumes a reference's already-optimized constants directly rather than
un-folding them.

The naive schedule it matches is *not* this package's `Poseidon` class, which
follows two different conventions: `Poseidon` S-boxes the last lane in a partial
round and orders each round `ARC -> S-box -> MDS`, whereas `SparsePoseidon`
S-boxes lane 0 and orders `S-box -> ARC(next) -> matrix` (the S-box precedes the
constant; the initial ARC seeds the first). The equivalence is to a naive Hades
with *those* conventions; `testing/sparse_test.py` pins it by deriving the sparse
factorization from a random naive instance and byte-matching.

Like `Poseidon`, the whole permute is one straight-line function (rounds
unrolled, linear layers via the normal-form helpers, no reduce/dot/gather) so it
lowers to a single kernel. The dense full-field matrices are too large for int64
literals, so — exactly as `Poseidon2`'s free-form (non-M4) path does — it rides
them as field arrays under the generic `zorch.fused_region` marker rather than a
name-routed emitter ABI. A dedicated `zorch.poseidon`-family emitter that
exploits the sparse structure (and its marker) is a perf follow-up; the generic
marker already fuses this body to one kernel meanwhile.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array

from zorch.fusion import FUSED_REGION_MARKER, fused_region
from zorch.hash.linear import apply_matrix
from zorch.hash.poseidon.linear import apply_sparse_partial
from zorch.hash.poseidon.params import SparsePoseidonParams

if TYPE_CHECKING:
    from zorch.hash.permutation import Permutation


class SparsePoseidon:
    """An optimized-sparse Poseidon permutation built from a SparsePoseidonParams;
    implements Permutation.

    permute = initial ARC -> (half-1) full rounds -> transition round (P)
              -> n_partial sparse rounds -> (half-1) full rounds
              -> final full round (no trailing constant), as one fused region.
    """

    def __init__(self, params: SparsePoseidonParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # False: the dense full-field matrices ride as closed-over field arrays
        # under the generic marker (a Goldilocks entry overflows an int64 literal,
        # so they cannot be a name-routed emitter's int attrs), and there is no
        # sparse-structure emitter to expand a whole region — so consumers keep it
        # on the per-block path, exactly like Poseidon2's free-form matrix.
        self.has_dedicated_fusion = False

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux seat
        # in `DuplexTranscript` (docs/reference/conventions.md "Pytree
        # registration").
        if not isinstance(other, SparsePoseidon):
            return NotImplemented
        return self._p == other._p

    def __hash__(self) -> int:
        return hash(self._p)

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got "
                f"{state.shape}"
            )
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must match the permutation field "
                f"{self.dtype}"
            )
        return _permute_body(self, state)

    # Inert fused-region ABI: non-dedicated, so consumers never route a
    # whole-region composite through this. A conformant stub (see
    # `Permutation.fused_region_spec`).
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        return (leading,), (lambda state, *ops: self.permute(state)), {}


def _permute_from_params(perm: "SparsePoseidon", s: Array) -> Array:
    """The optimized-sparse permute on a single `(width,)` state. Round constants
    ride as closed-over field arrays (added elementwise); the matrices apply via
    the field-array normal-form helpers. On the generic marker a closed-over
    constant lifts to an operand harmlessly."""
    p = perm._p
    alpha = p.alpha
    half = p.half_full_rounds
    mds = p.mds

    # S-box THEN add the round constant (the constant seeds the NEXT round's
    # S-box), then the linear layer — the constant follows the power map.
    def full_round(state: Array, rc: Array, matrix: Array) -> Array:
        return apply_matrix(matrix, fnp.power(state, alpha) + rc)

    # Initial ARC seeds the first S-box (added before any power map).
    s = s + p.initial_arc
    # Pre-partial full rounds with the dense MDS.
    for r in range(half - 1):
        s = full_round(s, p.full_rc_pre[r], mds)
    # Transition full round: the linear layer is P, not M.
    s = full_round(s, p.transition_rc, p.transition_matrix)
    # Partial rounds: the S-box hits lane 0 only (that is what makes them cheap).
    for r in range(p.n_partial_rounds):
        a = fnp.power(s[0], alpha) + p.partial_rc[r]
        s = apply_sparse_partial(p.partial_dot[r], p.partial_col[r], a, s[1:])
    # Post-partial full rounds with the dense MDS.
    for r in range(half - 1):
        s = full_round(s, p.full_rc_post[r], mds)
    # Final full round: S-box + MDS, no trailing constant.
    s = apply_matrix(mds, fnp.power(s, alpha))
    return s


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes. The
# permutation is the static key, compared by value; `inline=True` splices the
# cached jaxpr into the enclosing trace, so the emitted module is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: SparsePoseidon, state: Array) -> Array:
    def decomposition(s: Array) -> Array:
        return _permute_from_params(perm, s)

    return fused_region(decomposition, state, name=FUSED_REGION_MARKER)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[Permutation] = SparsePoseidon
