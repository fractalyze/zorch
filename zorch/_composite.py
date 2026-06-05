"""composite_or_inline — emit a `jax.lax.composite` marker, or run it inline.

`lax.composite` lowers to `stablehlo.CompositeOp`; jaxlib builds predating the
fork's composite backport (fractalyze/jax#164, not yet in a published wheel)
lack it, so lowering a composite under `@jit` fails there — which forces every
marked path onto the assertion-heavy self-built jaxlib (~28× slower compile).
When `CompositeOp` is absent the decomposition runs inline instead: numerically
identical, only the fusion marker is dropped (the zkx rewriter then has nothing
to fuse, which matters only on the GPU fusion path — not CPU dev/test, not
correctness). Auto-retires once the composite-capable wheel ships.

This fallback is pure jaxlib-compat infrastructure, not marker-specific
semantics, so it lives here once. Every zorch composite marker routes through
it — `fused_region`, `constraint_eval`, and the name-routed markers that go via
`fused_region` (poseidon2, merkle commit, the sumcheck round body).
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

# The region's output: a single Array for a one-kernel marker, or a pytree of
# Arrays for a name-routed region a vendor expands (e.g. a whole-tree commit
# returning (root, layers)). Both paths below preserve it.
_Region = TypeVar("_Region")


def composite_or_inline(
    decomposition: Callable[..., _Region],
    *operands: Array,
    name: str,
    version: int = 0,
    **attrs: object,
) -> _Region:
    """Wrap `decomposition` in a `lax.composite` marker, or run it inline.

    With `stablehlo.CompositeOp` available, emits
    `lax.composite(decomposition, name, version)(*operands, **attrs)` so the region
    lowers to one fused kernel (or a vendor-expanded chain, for a name-routed
    marker). Absent it, runs `decomposition(*operands, **attrs)` directly: the
    marker is dropped, the result identical.

    `attrs` ride along as composite attributes — the structural metadata a
    recognizing emitter parses (e.g. a sumcheck's `degree`/`num_vars`) — and are
    passed to `decomposition` on both paths, since `lax.composite` already calls
    the decomposition with them when tracing; so a decomposition whose attrs are
    required keyword arguments works inline too. `version` rides as
    `composite.version` for a recognizer that gates on a marker revision (default
    0, jax's default).
    """
    if not _HAS_COMPOSITE_OP:
        return decomposition(*operands, **attrs)
    return lax.composite(decomposition, name=name, version=version)(*operands, **attrs)
