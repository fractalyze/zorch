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
lowers to a single kernel under the generic `zorch.fused_region` marker.

There is also a dedicated `zorch.sparse_poseidon` name-routed marker — mirroring
`zorch.poseidon2` — whose emitter (`SparsePoseidonFusion`, in the Fractalyze XLA
plugin) exploits the sparse structure: the schedule shape and the four matrices
(`mds`, `transition_matrix`, and the per-round `partial_dot` / `partial_col`
pairs) ride as int64 marker attributes, the additive round constants as operands.
Two conditions gate that path: the pinned plugin has to ship the emitter
(`_DEDICATED_EMITTER_AVAILABLE`) and the instance's matrices have to fit the int64
attributes. Otherwise the permutation falls back to the generic marker (which
fuses this body to one kernel just the same) — emitting a marker the plugin cannot
recognize fails every compile with `custom op 'stablehlo.composite' is unknown`,
and a matrix over a field wider than an int64 (Goldilocks) has nothing the
attribute can carry. Either way the dedicated path's attributes and reference body
are exercised directly by `testing/sparse_test.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from zorch.fusion import FUSED_REGION_MARKER, fused_region
from zorch.hash.linear import apply_matrix
from zorch.hash.poseidon.linear import (
    apply_dense_mds,
    apply_sparse_partial,
    apply_sparse_partial_ints,
)
from zorch.hash.poseidon.params import SparsePoseidonParams

if TYPE_CHECKING:
    from zorch.hash.permutation import Permutation

POSEIDON_SPARSE_MARKER = "zorch.sparse_poseidon"
# Marker revision riding as `composite.version`. XLA recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker.
POSEIDON_SPARSE_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated
# `SparsePoseidonFusion` emitter. When True, `permute` emits the dedicated
# `zorch.sparse_poseidon` marker; when False it falls back to the generic
# `zorch.fused_region` marker (which fuses the same body to one kernel), because
# emitting a marker the plugin cannot recognize fails every compile with
# `custom op 'stablehlo.composite' is unknown`.
_DEDICATED_EMITTER_AVAILABLE = True

# Bounds of the dedicated marker's int64 matrix attributes. A field whose
# canonical values leave this range cannot ride that contract at all — see
# `_select_fused_region_name`.
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


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
        # Canonical-int views of the four matrices — the dedicated emitter carries
        # them as int64 marker attributes and the reference body applies them via
        # integer literals (no captured field array, which the name-routed marker
        # would lift to a leading operand). Extracted once here (eager), as classic
        # Poseidon does, because the marker choice below reads them too.
        rows = (
            params.mds_rows,
            params.transition_matrix_rows,
            params.partial_dot_rows,
            params.partial_col_rows,
        )
        self.fused_region_name = self._select_fused_region_name(rows)
        # Dedicated == permute lowers to the hash-named marker, not the generic
        # region one. Derived from the marker choice itself so the two can't drift
        # (mirrors Poseidon2).
        self.has_dedicated_fusion = self.fused_region_name != FUSED_REGION_MARKER
        self.fused_region_version = (
            POSEIDON_SPARSE_MARKER_VERSION if self.has_dedicated_fusion else 0
        )
        if self.has_dedicated_fusion:
            (
                self._mds_rows,
                self._transition_rows,
                self._partial_dot_rows,
                self._partial_col_rows,
            ) = rows

    def _select_fused_region_name(
        self, rows: tuple[tuple[tuple[int, ...], ...], ...]
    ) -> str:
        """Route to the dedicated `SparsePoseidonFusion` when the pinned plugin
        ships it AND this instance's matrices fit its int64 attribute contract,
        else the generic marker so compiles don't fail on an unknown composite or
        an unrepresentable attribute.

        Two gates, because readiness has two independent halves: the emitter's
        existence (`_DEDICATED_EMITTER_AVAILABLE`) and a data property, like
        Poseidon2's M4 check. Unlike Poseidon2 — whose only matrix attribute is
        the small structural M4, its full-width constants riding as operands —
        every linear layer here is a matrix of field elements, so a field whose
        canonical values exceed an int64 (Goldilocks, `p = 2^64 - 2^32 + 1`) has
        nothing the attribute can carry. Widening that contract is an emitter-side
        change (a u64 bit-cast and a version bump), so until then such a field
        takes the generic marker, which fuses the same body to one kernel."""
        if _DEDICATED_EMITTER_AVAILABLE and all(_fits_i64(m) for m in rows):
            return POSEIDON_SPARSE_MARKER
        return FUSED_REGION_MARKER

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux seat
        # in `DuplexTranscript` (docs/reference/conventions.md "Pytree
        # registration"). The marker name joins the key because it is part of what
        # `permute` lowers to and is not derived from the params alone: its int64
        # half is, but `_DEDICATED_EMITTER_AVAILABLE` is not. Without it a
        # dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache. In production that flag is a global
        # constant, so every live instance shares it and pytree-aux stability holds.
        if not isinstance(other, SparsePoseidon):
            return NotImplemented
        return self._p == other._p and self.fused_region_name == other.fused_region_name

    def __hash__(self) -> int:
        return hash((self._p, self.fused_region_name))

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

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The SparsePoseidonFusion ABI: operands `(leading, *additive round
        constants)`, the int-literal-matrix permute, and attrs whose four matrices
        (`mds` / `transition_matrix` / `partial_dot` / `partial_col`) name the
        linear layers. Dedicated path only; otherwise an inert stub (non-dedicated,
        so consumers never route a whole-region composite through it)."""
        if not self.has_dedicated_fusion:
            return (leading,), (lambda state, *ops: self.permute(state)), {}
        attrs, _version = _marker_attrs(self)
        return (
            _abi_operands(self, leading),
            partial(_permute_from_operands, self),
            {"permutation": "sparse_poseidon", **attrs},
        )


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


def _permute_from_operands(
    perm: "SparsePoseidon",
    s: Array,
    initial_arc: Array,
    full_rc_pre: Array,
    transition_rc: Array,
    partial_rc: Array,
    full_rc_post: Array,
) -> Array:
    """The SparsePoseidonFusion ABI reference body: the same schedule as
    `_permute_from_params`, but the additive round constants arrive as explicit
    operands (so `frx.lax.composite` can't lift them and break the ABI) and the
    four matrices apply via integer literals from `perm._*_rows` (so no field array
    is captured — the sparse structure rides as int64 marker attributes instead).
    `full_rc_pre` / `full_rc_post` arrive flattened row-major."""
    p = perm._p
    alpha = p.alpha
    w = perm.width
    half = p.half_full_rounds
    pre = full_rc_pre.reshape(half - 1, w)
    post = full_rc_post.reshape(half - 1, w)

    # S-box THEN add the round constant, then the int-literal linear layer.
    def full_round(state: Array, rc: Array, rows: tuple[tuple[int, ...], ...]) -> Array:
        return apply_dense_mds(rows, fnp.power(state, alpha) + rc)

    s = s + initial_arc
    for r in range(half - 1):
        s = full_round(s, pre[r], perm._mds_rows)
    # Transition full round: the linear layer is P, not M.
    s = full_round(s, transition_rc, perm._transition_rows)
    for r in range(p.n_partial_rounds):
        a = fnp.power(s[0], alpha) + partial_rc[r]
        s = apply_sparse_partial_ints(
            perm._partial_dot_rows[r], perm._partial_col_rows[r], a, s[1:]
        )
    for r in range(half - 1):
        s = full_round(s, post[r], perm._mds_rows)
    s = apply_dense_mds(perm._mds_rows, fnp.power(s, alpha))
    return s


def _abi_operands(perm: "SparsePoseidon", state: Array) -> tuple[Array, ...]:
    """The SparsePoseidonFusion ABI operands: `state` then the additive round
    constants (the transition round's `P` and the partial rounds' sparse pairs
    ride as attributes, not operands). `state` is `(width,)` single or
    `(n, width)` batched; the constants are identical either way. The full-round
    constants flatten row-major."""
    p = perm._p
    return (
        state,
        p.initial_arc,
        p.full_rc_pre.reshape(-1),
        p.transition_rc,
        p.partial_rc,
        p.full_rc_post.reshape(-1),
    )


def _fits_i64(rows: tuple[tuple[int, ...], ...]) -> bool:
    """Whether every canonical entry is representable as an int64 marker attribute.
    Checked before `_rows_to_i64` builds one, which would otherwise raise
    `OverflowError: Python int too large to convert to C long` at construction."""
    return all(_I64_MIN <= v <= _I64_MAX for row in rows for v in row)


def _rows_to_i64(rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Canonical-int rows flattened row-major as a numpy int64 array — a numpy
    value (not a Python list) so the marker attribute lowers to a
    `dense<[..]> : tensor<Nxi64>` the recognizer parses, not an unparsed ArrayAttr.
    Mirrors `Poseidon._poseidon_marker_attrs`' `mds`; the emitter supports only
    fields whose canonical values fit an int64 literal, which
    `_select_fused_region_name` has already established for a dedicated perm."""
    return np.array(rows, dtype=np.int64).flatten()


def _marker_attrs(perm: "SparsePoseidon") -> tuple[dict[str, object], int]:
    """The dedicated marker's `composite.attributes` + version: the schedule shape
    ints (the recognizer maps `alpha` to its s-box degree) plus the four matrices
    flattened row-major as int64 dense attrs. The marker NAME disambiguates the
    permutation, so no `permutation` discriminator here (that rides only in
    `fused_region_spec`, where a region could wrap any permutation). The body
    ignores these (metadata only); the generic marker stays attrs-free."""
    if not perm.has_dedicated_fusion:
        return {}, 0
    p = perm._p
    attrs: dict[str, object] = {
        "width": perm.width,
        "half_full_rounds": p.half_full_rounds,
        "n_partial_rounds": p.n_partial_rounds,
        "alpha": p.alpha,
        "mds": _rows_to_i64(perm._mds_rows),
        "transition_matrix": _rows_to_i64(perm._transition_rows),
        "partial_dot": _rows_to_i64(perm._partial_dot_rows),
        "partial_col": _rows_to_i64(perm._partial_col_rows),
    }
    return attrs, POSEIDON_SPARSE_MARKER_VERSION


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes. The
# permutation is the static key, compared by value; `inline=True` splices the
# cached jaxpr into the enclosing trace, so the emitted module is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: SparsePoseidon, state: Array) -> Array:
    # `perm` is static, so this branch resolves at trace time (no HLO select).
    if not perm.has_dedicated_fusion:

        def decomposition(s: Array) -> Array:
            return _permute_from_params(perm, s)

        return fused_region(decomposition, state, name=FUSED_REGION_MARKER)

    def dedicated(
        s: Array,
        initial_arc: Array,
        full_rc_pre: Array,
        transition_rc: Array,
        partial_rc: Array,
        full_rc_post: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata (the shape ints + matrices); the body
        # ignores it and applies the matrices via `perm._*_rows` int literals.
        return _permute_from_operands(
            perm, s, initial_arc, full_rc_pre, transition_rc, partial_rc, full_rc_post
        )

    attrs, version = _marker_attrs(perm)
    return fused_region(
        dedicated,
        *_abi_operands(perm, state),
        name=perm.fused_region_name,
        version=version,
        **attrs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[Permutation] = SparsePoseidon
