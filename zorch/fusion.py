"""fused_region — mark a straight-line region as one fused kernel.

Wraps a decomposition in a `frx.lax.composite` named `zorch.fused_region`; FXLA's
`ZorchFusedRegionRewriter` turns that marker into a single custom-fusion kernel —
one kernel by construction, not by a per-hash compiler pattern match. The name is
deliberately generic: one marker fuses any straight-line region (a Round, a fold,
a hash permutation, …), so it is not named after any single use — see CLAUDE.md.

The decomposition must be straight-line element-wise — no loops, reductions, or
gathers — so the region lowers to one kernel: a round sequence is unrolled into
the body (fixed, small counts) and the linear layers use the normal-form helpers
(not `jnp.dot`/`reduce`/`gather`). Use `lax` primitives over `jnp` wrappers
(`lax.select`, not `jnp.where`): a `jnp` wrapper's internal `jit` lowers to a
call inside the body, which the single-kernel rewriter rejects. Name-routed
markers with a dedicated emitter (`zorch.sumcheck`, `zorch.poseidon2`) are exempt —
their emitters tolerate reductions and calls. Loop-carrying large-N rounds
await the in-kernel-loop emitter; see fractalyze/zorch#25.

The `lax.composite` marker is emitted via `zorch._composite.composite`, the one
place every zorch composite marker routes through.
"""

from __future__ import annotations

from collections.abc import Callable

from frx import Array

from zorch._composite import _Region, composite

FUSED_REGION_MARKER = "zorch.fused_region"


def fused_region(
    decomposition: Callable[..., _Region],
    *operands: Array,
    name: str = FUSED_REGION_MARKER,
    version: int = 0,
    **attrs: object,
) -> _Region:
    """Mark a region (`decomposition`) as one fused kernel — or, under a
    name-routed marker, a boundary a vendor expands into a kernel chain.

    Under the default marker the region must be straight-line element-wise — no
    loops, reductions, or gathers — so it lowers to a single kernel. It is called
    with `operands`, which become the composite's operands in order.

    A non-default `name` routes the region to a dedicated FXLA emitter instead of
    the generic one — e.g. a `zorch.poseidon2` region goes to `Poseidon2Fusion`
    rather than the generic `LoopFusion` (unusable for a full hash permutation).
    The `operands` must then follow that emitter's ABI. Such a region need not be
    single-kernel: a vendor may expand it into a kernel chain, so `decomposition`
    may return a pytree, not just an Array.

    `version` and `attrs` ride through to the composite (`composite.version` and
    `composite.attributes`) — the structural metadata a recognizer parses, e.g. a
    sumcheck marker's `degree` / `num_vars`. Both default to absent (`version=0`,
    no attrs), so a plain straight-line region is unchanged.
    """
    return composite(decomposition, *operands, name=name, version=version, **attrs)
