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
optimize) -- a different fusion shape. ``assert_input_uses`` and
``primitive_count`` are scheme-agnostic traced-graph accounting, so they apply
to any body whose trace size matters -- a hash layer, a sumcheck round, a fold
step.
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


def assert_input_uses(
    fn: Callable[..., Any], *args: Any, arg: int = 0, limit: int
) -> None:
    """Assert ``fn`` reads its ``arg``-th traced input at most ``limit`` times.

    A body must not read a traced input more times than its output shape
    requires. Each extra read is a separate expression the compiler re-derives
    the value for, so where the value is loop-carried -- a permutation's rounds,
    a fold's iterations -- the cost compounds per iteration rather than adding.
    ``zorch.hash.linear`` tells the story that produced this rule.

    ``limit`` is where the scaling law goes: pass ``limit=width`` for a layer
    whose lanes each legitimately touch the state once, or the measured constant
    for a value every lane shares. Written that way one call already pins the
    scaling; a second width only catches a superlinearity whose constant is
    small enough to hide at the first.
    """
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    var = jaxpr.invars[arg]
    n = sum(eqn.invars.count(var) for eqn in jaxpr.eqns)
    if n > limit:
        raise AssertionError(
            f"input {arg} read {n} times, over the limit of {limit}. Read it once "
            f"as an array, or hoist its lanes once and index the hoisted list."
        )


def primitive_count(fn: Callable[..., Any], *args: Any, name: str) -> int:
    """How many ``name`` primitives ``fn`` traces to -- the same traced-graph
    accounting as ``assert_input_uses``, counting ops instead of reads."""
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    return sum(1 for eqn in jaxpr.eqns if eqn.primitive.name == name)
