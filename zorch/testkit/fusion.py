# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only fusion-readiness assertion for straight-line IOP round bodies.

A round body (sumcheck, logup-gkr) must lower to element-wise field ops + the
one inherent Sigma -- no gather/scatter/dot/while/... boundary, no extra reduce.
``assert_fusion_ready`` lowers ``fn(*args)`` and checks the StableHLO uses only
fusion-safe ops plus exactly ``reduces`` reduce(s). It's a whitelist (not a
gather/dot blacklist), so ANY boundary op or extra reduce trips it -- and any
new op in the fusion-critical body gets a conscious look. Cheap proxy for XLA's
``ZorchFusedRegionRewriter``, the authoritative compiler gate.

``assert_fusion_ready`` is not for the hash permutation: poseidon2 fuses via the
``zorch.fused_region`` marker and normal-form linear layers (no dot for XLA to
optimize) -- a different fusion shape, whose assertion is
``assert_marker_recognized``.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import frx

# Element-wise field + structural ops that stay inside one kernel.
_FUSION_SAFE = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "negate",
        "constant",
        "convert",
        "broadcast_in_dim",
        "reshape",
        "slice",
        "concatenate",
        "transpose",
    }
)


def assert_fusion_ready(fn: Callable[..., Any], *args: Any, reduces: int = 0) -> None:
    """Assert ``fn``'s lowered body is straight-line element-wise plus exactly
    ``reduces`` reduce(s); raise ``AssertionError`` naming offenders otherwise."""
    hlo = frx.jit(fn).lower(*args).as_text()
    ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
    n = ops.count("reduce")
    if n != reduces:
        raise AssertionError(
            f"expected {reduces} reduce(s), got {n} (ops: {sorted(set(ops))})"
        )
    offenders = sorted({o for o in ops if o != "reduce" and o not in _FUSION_SAFE})
    if offenders:
        raise AssertionError(f"non-fusion-safe ops in body: {offenders}")


def custom_fusion_names(fn: Callable[..., Any], *args: Any) -> list[str]:
    """The routing keys of ``fn``'s compiled custom fusions, in module order.

    The compiled module is the only place a RECOGNIZED marker is visible.
    Emitting a marker and having a vendor emitter route it are different
    properties, and the difference does not show in the output: an unrecognized
    name is not an error, it inlines back to the decomposition and computes
    identical bytes. So every value-level test passes either way, and a lowered
    module (``.lower(...).as_text()``) proves only that zorch wrote the string.
    """
    compiled = frx.jit(fn).lower(*args).compile().as_text()
    names = []
    for line in compiled.splitlines():
        if "kind=kCustom" not in line:
            continue
        # `ROOT ` prefixes the entry computation's own fusion, and XLA appends a
        # `.N` disambiguator once a name repeats -- neither changes the key.
        m = re.match(
            r"\s*(?:ROOT\s+)?%([A-Za-z0-9_-]+(?:\.[A-Za-z_-][A-Za-z0-9_-]*)*)", line
        )
        if m:
            names.append(m.group(1))
    return names


def assert_marker_recognized(
    routing_key: str, fn: Callable[..., Any], *args: Any
) -> None:
    """Assert ``fn`` compiles to a custom fusion named ``routing_key``.

    Marker names are a wire ABI shared with Fractalyze XLA, so this is what
    catches a rename here that the pinned toolchain does not accept, and a
    toolchain bump that drops a name zorch still emits. The instruction name is
    matched whole rather than by substring: ``poseidon`` is a prefix of
    ``poseidon2``, so a substring match would let either emitter satisfy the
    other's assertion.
    """
    names = custom_fusion_names(fn, *args)
    if not names:
        raise AssertionError(f"no custom fusion at all: {routing_key} is unrecognized")
    if routing_key not in names:
        raise AssertionError(f"recognized, but not as {routing_key}: {names}")
