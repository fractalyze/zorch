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

from jax import Array, lax

try:
    from jaxlib.mlir.dialects import stablehlo as _stablehlo

    _HAS_COMPOSITE_OP = hasattr(_stablehlo, "CompositeOp")
except ImportError:  # pragma: no cover - jaxlib MLIR bindings unavailable
    _HAS_COMPOSITE_OP = False

FUSED_REGION_MARKER = "zorch.fused_region"


def fused_region(
    decomposition: Callable[[Array], Array],
    init: Array,
    *,
    name: str = FUSED_REGION_MARKER,
) -> Array:
    """Mark a straight-line region (`decomposition`) as one fused kernel.

    `decomposition` must be straight-line element-wise — no loops, reductions, or
    gathers — so the marked region lowers to a single kernel. On a jaxlib without
    `stablehlo.CompositeOp` the marker is dropped and `decomposition` runs inline
    (see the module docstring).
    """
    if not _HAS_COMPOSITE_OP:
        return decomposition(init)
    return lax.composite(decomposition, name=name)(init)
