# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Cached `jax.export` dispatch for the jagged sumcheck round kernels.

A shape-polymorphic exported round binary is built once per (round-kind,
dtype-mix) and re-dispatched at every concrete round size, layer, and shard —
the compile surface is O(1) in round count and shard count, where tracing the
round loop per layer shape pays a fresh trace + XLA compile for every layer of
every distinct shard height profile.

Two caches with distinct coverage:

- This module's cache holds the **exported binaries** (the symbolic StableHLO
  build). In-memory always; opt-in on-disk via ``ZORCH_EXPORT_CACHE_DIR`` so
  the export build survives across processes. The XLA persistent compile cache
  does NOT cover this stage.
- ``exported.call`` still re-runs **XLA codegen per concrete operand shape**
  (structural: the refined module hash is the compile key). Point
  ``JAX_COMPILATION_CACHE_DIR`` at a persistent directory so those executables
  survive across proves and processes; without it a deep pyramid's hundreds of
  distinct halving sizes re-codegen every run and the binaries never warm.

Soundness of the on-disk cache: a stale binary silently emits the OLD
arithmetic — a wrong proof. The cache directory is therefore namespaced by the
jax version plus a hash of **every** ``zorch`` source file, not a hand-picked
import list (an enumerated subset once missed the summand and interpolation
modules and served stale binaries). Whole-package hashing over-invalidates
slightly; it can never serve stale math.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
from jax import export

# `Exported.call` wraps every invocation in a `custom_vjp` (its AD path), which
# the eager host round loop never differentiates; binding the primitive
# directly is ~177us -> ~55us warm per dispatch with the same binary and the
# same flat operands — byte-identical.
from jax._src.export._export import call_exported_p as _call_exported_p

_BINARY_CACHE: dict[tuple, export.Exported] = {}

_CACHE_DIR_ENV = "ZORCH_EXPORT_CACHE_DIR"


def _package_source_hash() -> str:
    """Hash of every .py under the zorch package — the coarse closure key."""
    root = Path(__file__).parents[1]
    h = hashlib.sha256()
    for src in sorted(root.rglob("*.py")):
        h.update(bytes(src.relative_to(root)))
        h.update(src.read_bytes())
    return h.hexdigest()[:12]


_UNRESOLVED = object()
_disk_dir_cache: Any = _UNRESOLVED


def _disk_dir() -> Path | None:
    """The namespaced on-disk cache dir; None when the env var is unset or the
    serialization backend is unavailable (resolved once per process)."""
    global _disk_dir_cache
    if _disk_dir_cache is _UNRESOLVED:
        base = os.environ.get(_CACHE_DIR_ENV)
        if base is None:
            _disk_dir_cache = None
        else:
            try:
                import flatbuffers  # noqa: F401  # Exported.(de)serialize backend
            except ImportError:
                warnings.warn(
                    f"{_CACHE_DIR_ENV} is set but 'flatbuffers' is not installed; "
                    "exported round binaries stay in-memory only",
                    stacklevel=2,
                )
                _disk_dir_cache = None
            else:
                d = Path(base) / f"{jax.__version__}-{_package_source_hash()}"
                d.mkdir(parents=True, exist_ok=True)
                _disk_dir_cache = d
    return _disk_dir_cache


def _disk_path(key: tuple) -> Path | None:
    d = _disk_dir()
    if d is None:
        return None
    return d / f"{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.bin"


def dispatch(key: tuple, operands: tuple, build: Callable[[], export.Exported]) -> Any:
    """Call the exported binary cached under ``key`` on ``operands``, building
    (and caching) it on a miss. The symbolic export is the cold cost, so
    ``build`` runs only on a miss; the call binds ``call_exported_p`` directly
    (see module docstring)."""
    exported = _BINARY_CACHE.get(key)
    if exported is None:
        path = _disk_path(key)
        if path is not None and path.exists():
            exported = export.deserialize(bytearray(path.read_bytes()))
        else:
            exported = build()
            if path is not None:
                # Atomic publish: write a per-pid sibling temp then os.replace
                # into place, so a process sharing the cache dir never
                # deserializes a half-written .bin (rename is atomic within
                # one filesystem).
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_bytes(bytes(exported.serialize()))
                os.replace(tmp, path)
        _BINARY_CACHE[key] = exported
    flat = jax.tree_util.tree_leaves(operands)
    return exported.out_tree.unflatten(_call_exported_p.bind(*flat, exported=exported))


def register_operand_pytrees(*types: type) -> None:
    """Register operand pytree dataclasses (no meta fields) for
    ``Exported.serialize`` so the on-disk cache can round-trip binaries whose
    signatures carry them. Idempotent across re-imports."""
    for t in types:
        try:
            export.register_pytree_node_serialization(
                t,
                serialized_name=f"{t.__module__}.{t.__name__}",
                serialize_auxdata=lambda _a: b"",
                deserialize_auxdata=lambda _b: (),
            )
        except ValueError:
            # importlib.reload re-runs the registration with the same constant
            # serialized_name; the prior, identical registration is live.
            pass
