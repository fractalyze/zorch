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
"""

from __future__ import annotations

from collections.abc import Callable

from jax import Array, lax

FUSED_REGION_MARKER = "zorch.fused_region"


def fused_region(
    decomposition: Callable[[Array], Array],
    init: Array,
    *,
    name: str = FUSED_REGION_MARKER,
) -> Array:
    """Mark a straight-line region (`decomposition`) as one fused kernel.

    `decomposition` must be straight-line element-wise — no loops, reductions, or
    gathers — so the marked region lowers to a single kernel.
    """
    return lax.composite(decomposition, name=name)(init)
