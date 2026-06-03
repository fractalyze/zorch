# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only fusion-readiness assertion for straight-line IOP round bodies.

A round body (sumcheck, logup-gkr) must lower to element-wise field ops + the
one inherent Sigma -- no gather/scatter/dot/while/... boundary, no extra reduce.
``assert_fusion_ready`` lowers ``fn(*args)`` and checks the StableHLO uses only
fusion-safe ops plus exactly ``reduces`` reduce(s). It's a whitelist (not a
gather/dot blacklist), so ANY boundary op or extra reduce trips it -- and any
new op in the fusion-critical body gets a conscious look. Cheap proxy for zkx's
``ZorchFusedRegionRewriter`` (issue #21), the authoritative compiler gate.

Not for the hash permutation: poseidon2 fuses via the ``zorch.fused_region``
marker and normal-form linear layers (no dot for zkx to optimize) -- a different
fusion shape.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import jax

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
    hlo = jax.jit(fn).lower(*args).as_text()
    ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
    n = ops.count("reduce")
    if n != reduces:
        raise AssertionError(
            f"expected {reduces} reduce(s), got {n} (ops: {sorted(set(ops))})"
        )
    offenders = sorted({o for o in ops if o != "reduce" and o not in _FUSION_SAFE})
    if offenders:
        raise AssertionError(f"non-fusion-safe ops in body: {offenders}")
