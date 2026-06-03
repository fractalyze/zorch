"""fused_region — mark a straight-line region as one fused kernel.

Wraps a decomposition in a `jax.lax.composite` named `zorch.fused_region`; zkx's
`ZorchFusedRegionRewriter` turns that marker into a single custom-fusion kernel —
one kernel by construction, not by a per-hash compiler pattern match. The name is
deliberately generic: one marker fuses any straight-line region (a Round, a fold,
a hash permutation, …), so it is not named after any single use — see CLAUDE.md.

The decomposition must be straight-line element-wise — no loops, reductions, or
gathers — so the region lowers to one kernel: a round sequence is unrolled into
the body (fixed, small counts) and the linear layers use the normal-form helpers
(not `jnp.dot`/`reduce`/`gather`). Loop-carrying large-N rounds await the
in-kernel-loop emitter; see fractalyze/zorch#25.

`lax.composite` lowers to `stablehlo.CompositeOp`; jaxlib builds predating the
fork's composite backport (fractalyze/jax#164, not yet in a published wheel) lack
it, so lowering a composite under `@jit` fails there — which forces every
Poseidon2/Round/fold path onto the assertion-heavy self-built jaxlib (~28× slower
compile). When `CompositeOp` is absent, `fused_region` runs the decomposition
inline instead: numerically identical, only the fusion marker is dropped (the
`ZorchFusedRegionRewriter` then has nothing to fuse, which matters only on the GPU
fusion path — not CPU dev/test, not correctness). Auto-retires once the
composite-capable wheel ships.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from jax import Array, lax

try:
    from jaxlib.mlir.dialects import stablehlo as _stablehlo

    _HAS_COMPOSITE_OP = hasattr(_stablehlo, "CompositeOp")
except ImportError:  # pragma: no cover - jaxlib MLIR bindings unavailable
    _HAS_COMPOSITE_OP = False

FUSED_REGION_MARKER = "zorch.fused_region"

# The region's output: a single Array for a one-kernel marker, or a pytree of
# Arrays for a name-routed region a vendor expands (e.g. a whole-tree commit
# returning (root, layers)). lax.composite preserves it either way.
_Region = TypeVar("_Region")


def fused_region(
    decomposition: Callable[..., _Region],
    *operands: Array,
    name: str = FUSED_REGION_MARKER,
) -> _Region:
    """Mark a region (`decomposition`) as one fused kernel — or, under a
    name-routed marker, a boundary a vendor expands into a kernel chain.

    Under the default marker the region must be straight-line element-wise — no
    loops, reductions, or gathers — so it lowers to a single kernel. It is called
    with `operands`, which become the composite's operands in order. On a jaxlib
    without `stablehlo.CompositeOp` the marker is dropped and `decomposition` runs
    inline (see the module docstring).

    A non-default `name` routes the region to a dedicated zkx emitter instead of
    the generic one — e.g. a `poseidon2:W:E:I:S` region goes to `Poseidon2Fusion`
    rather than the generic `LoopFusion` (unusable for a full hash permutation).
    The `operands` must then follow that emitter's ABI. Such a region need not be
    single-kernel: a vendor may expand it into a chain (e.g. `zorch.merkle_commit`
    → per-layer hash kernels), so `decomposition` may return a pytree, not just an
    Array.
    """
    if not _HAS_COMPOSITE_OP:
        return decomposition(*operands)
    return lax.composite(decomposition, name=name)(*operands)
