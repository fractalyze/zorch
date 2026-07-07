# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-round `jax.export` machinery: the on-disk/in-memory binary
cache keyed by the round arithmetic's source hash, and the single-round
symbolic-shape dispatches."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import jax
from jax import Array, export
from jax._src.export._export import call_exported_p as _call_exported_p

from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_types import (
    _InterpConsts,
    _Planes,
    _RoundScalars,
)

# Exported per-round kernels, keyed by the operand signature so one binary
# serves every round size in its bracket and is reused across rounds, layers, and
# shards (the recompile-free dispatch). Only the per-round-REPEATED variants are
# cached here -- `fix_and_sum_row` (the row rounds) and `fix_and_sum_int` (the
# interaction rounds); `sum_as_poly_row` / `_boundary` / `_last` fire once a layer
# and stay eager. The state dtype is part of the key: a multi-limb sumcheck folds
# base->extension after round 0, so the row binary is dispatched at two input
# dtypes (numerator base-field, denominator extension-field).
#
# The bare `Exported` is cached, not `jax.jit(exported.call)`: jit-wrapping cuts
# the per-call host dispatch (cached exec vs bare's per-call re-specialize) but is
# wall-clock NEUTRAL on the real prove -- the round dispatch overlaps async GPU/FS
# work, so the saved host time is off the critical path. Not worth the extra layer.
_ROUND_KERNEL_CACHE: dict[tuple, export.Exported] = {}

# Opt-in on-disk cache for the exported round binaries (set ZORCH_EXPORT_CACHE_DIR):
# their jax.export BUILD (symbolic StableHLO generation) re-runs every process and
# dominates the cold start, which the XLA persistent compile cache does NOT cover.
# Namespaced by jax version + a hash of every module the exported kernels close
# over (see `_export_cache_dir`), so any kernel-arithmetic edit invalidates it.
# Unset -> unchanged in-memory behaviour.
_EXPORT_CACHE_DIR = os.environ.get("ZORCH_EXPORT_CACHE_DIR")


@cache
def _export_cache_dir() -> Path:
    import hashlib

    import zorch.logup_gkr.circuit as _circuit
    import zorch.logup_gkr.prover as _prover
    import zorch.poly.eq as _eq
    import zorch.poly.univariate as _univariate

    # Hash every module whose code the exported round kernels close over, so any
    # edit to the round arithmetic invalidates the on-disk binary: the whole
    # jagged-prover module family (`_jagged_*.py` + `jagged_prover.py`), the
    # eq / circuit plane builders, `logup_combine` (the summand) in logup_gkr.prover,
    # and the Lagrange/Vandermonde interpolation in poly.univariate. Miss one and a
    # stale binary silently emits the OLD arithmetic — a wrong proof.
    h = hashlib.sha256()
    for mod in (_circuit, _eq, _prover, _univariate):
        src = mod.__file__
        assert src is not None  # imported modules always carry a source path
        h.update(Path(src).read_bytes())
    here = Path(__file__).resolve().parent
    for src_path in sorted(here.glob("_jagged_*.py")) + [here / "jagged_prover.py"]:
        h.update(src_path.read_bytes())
    d = (
        # Reached only under the `_EXPORT_CACHE_DIR is not None` guards in
        # `_round_get`/`_round_put`, so the env var is a real path string here.
        Path(_EXPORT_CACHE_DIR)  # type: ignore[arg-type]
        / f"{jax.__version__}-{h.hexdigest()[:12]}"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _export_path(key: tuple) -> Path:
    import hashlib

    return (
        _export_cache_dir()
        / f"{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.bin"
    )


def _round_get(key: tuple) -> export.Exported | None:
    exp = _ROUND_KERNEL_CACHE.get(key)
    if exp is None and _EXPORT_CACHE_DIR is not None:
        path = _export_path(key)
        if path.exists():
            exp = export.deserialize(bytearray(path.read_bytes()))
            _ROUND_KERNEL_CACHE[key] = exp
    return exp


def _round_put(key: tuple, exp: export.Exported) -> None:
    _ROUND_KERNEL_CACHE[key] = exp
    if _EXPORT_CACHE_DIR is not None:
        # Atomic publish: write a per-pid sibling temp then os.replace into place,
        # so a process sharing ZORCH_EXPORT_CACHE_DIR never deserializes a
        # half-written .bin (rename is atomic within one filesystem).
        path = _export_path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(bytes(exp.serialize()))
        os.replace(tmp, path)


def _round_dispatch(
    key: tuple,
    operands: tuple,
    build: Callable[[], export.Exported],
) -> Any:
    """The shared round export-cache dispatch every `_dispatch_*` runs: reuse the
    cached binary for `key`, else `build()` it (the symbolic export is the cold
    cost, so only on a miss) and cache it, then call it on the concrete `operands`.
    The tracer fallback, `operands`, `key`, and the abstract shapes stay per-round;
    only this get / export / put / call protocol is shared.

    The call binds `call_exported_p` directly rather than going through
    `Exported.call`: that method wraps every invocation in a `custom_vjp` (its
    `f_imported`/`f_flat` AD path), which the eager host round loop never
    differentiates. Skipping it is ~177us -> ~55us warm per dispatch (a `jax.jit`
    dispatch costs the same) on the dispatch-bound host-FS prove, with the SAME one
    symbolic binary -- no per-shape recompile, byte-identical (same primitive, same
    flat operands)."""
    exported = _round_get(key)
    if exported is None:
        exported = build()
        _round_put(key, exported)
    flat = jax.tree_util.tree_leaves(operands)
    return exported.out_tree.unflatten(_call_exported_p.bind(*flat, exported=exported))


# `_Planes` / `_RoundScalars` cross the jax.export boundary as pytree operands;
# register their (empty -- no meta_fields) aux so `Exported.serialize()` can
# round-trip them for the on-disk cache above. The serialized_name is pinned to
# the historical `jagged_prover` module path (the classes moved to
# `_jagged_types`): it is a wire identifier inside the cached binaries, and
# renaming it would orphan every existing on-disk cache entry.
if _EXPORT_CACHE_DIR is not None:
    for _t in (_Planes, _RoundScalars):
        try:
            export.register_pytree_node_serialization(
                _t,
                serialized_name=f"zorch.logup_gkr.jagged_prover.{_t.__name__}",
                serialize_auxdata=lambda _a: b"",
                deserialize_auxdata=lambda _b: (),
            )
        except ValueError:
            # Idempotent across a re-import (importlib.reload): serialized_name is
            # a constant, so jax raises "Duplicate serialization registration".
            # The prior, identical registration is already live -- keep it.
            pass

# The symbolic bound only needs to *contain* every round size (`exported.call`
# re-specializes XLA codegen per concrete size regardless), but it MUST exceed the
# largest dispatched state: a row input is `2*pp` and an interaction input `4*g`,
# so the bound caps the provable layer at `2*_ROUND_SYM_MAX` / `4*_ROUND_SYM_MAX`
# elements. Hold it well above any trace's 2^(log rows) so no real shard overflows.
_ROUND_SYM_MAX = 1 << 30

_SCALAR_FIELDS = ("eq_adj", "pad_adj", "z_cur", "claim", "lam")


def _abst_scalars(scalars: _RoundScalars) -> _RoundScalars:
    return _RoundScalars(
        *(jax.ShapeDtypeStruct((), getattr(scalars, f).dtype) for f in _SCALAR_FIELDS)
    )


def _dispatch_fix_and_sum_int(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the dense interaction round through one cached binary symbolic
    over the state size. The round is width-preserving (the folded state
    zero-pads back to the input width, live tracked by `live`), so the state
    and `eq_int` share one `4*g` symbol -- and under fixed caps (xla#179) `g`
    only ever binds one concrete size, so the binary specializes exactly once.

    `exported.call` is a host dispatch; under a `jax.jit` trace the operands are
    tracers, so fall back to the eager kernel -- the jit compiles the round
    itself, the per-round export being its alternative."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_dense(
            planes, eq_int, alpha, scalars, consts, live
        )
    operands = (planes, eq_int, alpha, scalars, live)
    # Per-operand dtypes (a LogUp numerator is base-field, its denominator
    # extension-field, and the state promotes base->extension across rounds), so
    # each (round-shape, dtype-mix) gets its own binary; `consts` is baked in.
    key = (
        "int",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )

    def build() -> export.Exported:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((4 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((2,), live.dtype),
        )
        fn = lambda p, e, al, sc, lv: _composite_fix_and_sum_dense(  # noqa: E731
            p, e, al, sc, consts, lv
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_boundary(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
) -> tuple[Array, _Planes, Array]:
    """Dispatch the row->interaction handoff (bind the last row challenge `alpha`,
    then sum the first interaction round over the still-unfolded `eq_int`) through
    one cached binary. Mirrors `_dispatch_fix_and_sum_int` without the `eq_int`
    bind: the bind halves the state (`4*g -> 2*g`) and `eq_int` rides unfolded at
    `2*g` (= the post-bind state), so one dispatched kernel replaces the eager one.
    """
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_boundary(
            planes, eq_int, alpha, scalars, consts, live
        )
    operands = (planes, eq_int, alpha, scalars, live)
    key = (
        "boundary",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        consts.naturals.shape[0],
        consts.naturals.dtype,
    )

    def build() -> export.Exported:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((2,), live.dtype),
        )
        fn = lambda p, e, al, sc, lv: _composite_fix_and_sum_boundary(  # noqa: E731
            p, e, al, sc, consts, lv
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_sum_as_poly_row(
    planes: _Planes,
    row_counts: Array,
    eq_row: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes]:
    """Dispatch the round-0 sum (no fold, no challenge) through one cached
    binary. The capped (width-preserving, `out_pairs` None) route exports
    symbolic over the raw height (`2*h2`) and `eq_row` (`2*rr`) — the derived
    schedule's width follows the plane width, so the marker v2 ABI needs no
    schedule symbol; `eq_int` and `row_counts` ride fixed. The exact route's
    padded width (`out_pairs`) is a static shape, so it exports concrete,
    keyed per layout. Mirrors `_dispatch_fix_and_sum_row` without the bind."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_sum_as_poly_row(
            planes, row_counts, eq_row, eq_int, scalars, consts, live, out_pairs
        )
    operands = (planes, row_counts, eq_row, eq_int, scalars, live)
    key = (
        "sum0",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        consts.naturals.shape[0],
        out_pairs,
        None if out_pairs is None else (planes.n0.shape, eq_row.shape),
    )

    def build() -> export.Exported:
        if out_pairs is None:
            h2, rr = export.symbolic_shape(
                "h2, rr",
                constraints=[
                    "h2 >= 1",
                    f"h2 <= {_ROUND_SYM_MAX}",
                    "rr >= 1",
                    f"rr <= {_ROUND_SYM_MAX}",
                ],
            )
            plane_w, eq_w = 2 * h2, 2 * rr
        else:
            plane_w, eq_w = planes.n0.shape[0], eq_row.shape[0]
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((plane_w,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct((eq_w,), eq_row.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((3,), live.dtype),
        )
        fn = lambda pl, rc, er, ei, sc, lv: _composite_sum_as_poly_row(  # noqa: E731
            pl, rc, er, ei, sc, consts, lv, out_pairs
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)


def _dispatch_fix_and_sum_row(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live: Array,
    out_pairs: int | None = None,
) -> tuple[Array, _Planes, Array]:
    """Dispatch a jagged row round through one cached binary. The capped
    (width-preserving, `out_pairs` None) route exports symbolic over the input
    state (`2*pp`) and the width-preserved `eq_row` (`2*rr`) — the derived
    schedule's width follows the plane width (marker v2), so no schedule
    symbol exists; `eq_int` and `row_counts` ride fixed, and under fixed caps
    every symbol binds one concrete size, so the binary specializes exactly
    once. The exact route's padded width (`out_pairs`) is a static shape, so
    it exports concrete, keyed per layout."""
    if isinstance(planes.n0, jax.core.Tracer):
        return _composite_fix_and_sum_row(
            planes,
            eq_row,
            alpha,
            row_counts,
            eq_int,
            scalars,
            consts,
            live,
            out_pairs,
        )
    operands = (
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        live,
    )
    key = (
        "row",
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        consts.naturals.shape[0],
        out_pairs,
        None if out_pairs is None else (planes.n0.shape, eq_row.shape),
    )

    def build() -> export.Exported:
        if out_pairs is None:
            pp, rr = export.symbolic_shape(
                "pp, rr",
                constraints=[
                    "pp >= 1",
                    f"pp <= {_ROUND_SYM_MAX}",
                    "rr >= 1",
                    f"rr <= {_ROUND_SYM_MAX}",
                ],
            )
            plane_w, eq_w = 2 * pp, 2 * rr
        else:
            plane_w, eq_w = planes.n0.shape[0], eq_row.shape[0]
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((plane_w,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((eq_w,), eq_row.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct((3,), live.dtype),
        )
        fn = (  # noqa: E731
            lambda pl, er, al, rc, ei, sc, lv: _composite_fix_and_sum_row(
                pl, er, al, rc, ei, sc, consts, lv, out_pairs
            )
        )
        return export.export(jax.jit(fn))(*abst)

    return _round_dispatch(key, operands, build)
