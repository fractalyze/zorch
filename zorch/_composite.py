"""composite — emit a `jax.lax.composite` marker for a fusible region.

`lax.composite` lowers to `stablehlo.CompositeOp`; zkx's rewriter then fuses the
marked region into a single custom-fusion kernel (or a vendor-expanded chain,
for a name-routed marker). Marker-specific semantics live in the callers
(`fused_region`, `constraint_eval`, the name-routed poseidon2/merkle/sumcheck
markers); this is the one place the `lax.composite` call is spelled.

`lax.composite` re-traces its decomposition on EVERY emission (no trace cache
in jax), so a hot identical-aval emission site pays the full Python body per
call. The cache deliberately does NOT live here: most call sites pass fresh
closures (identity keys would grow a cache unboundedly), and a shared cache
would need private jax internals. Instead, hoist the hot site into a
module-level value-keyed `jax.jit(..., inline=True)` zone — see
`zorch.hash.poseidon2.poseidon2._permute_body`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from jax import Array, lax

# The region's output: a single Array for a one-kernel marker, or a pytree of
# Arrays for a name-routed region a vendor expands (e.g. a whole-tree commit
# returning (root, layers)). Both paths below preserve it.
_Region = TypeVar("_Region")


def composite(
    decomposition: Callable[..., _Region],
    *operands: Array,
    name: str,
    version: int = 0,
    **attrs: object,
) -> _Region:
    """Wrap `decomposition` in a `lax.composite` marker.

    Emits `lax.composite(decomposition, name, version)(*operands, **attrs)` so the
    region lowers to one fused kernel (or a vendor-expanded chain, for a
    name-routed marker).

    `attrs` ride along as composite attributes — the structural metadata a
    recognizing emitter parses (e.g. a sumcheck's `degree`/`num_vars`) — and are
    passed to `decomposition`, since `lax.composite` calls the decomposition with
    them when tracing; so a decomposition whose attrs are required keyword
    arguments works. `version` rides as `composite.version` for a recognizer that
    gates on a marker revision (default 0, jax's default).
    """
    return lax.composite(decomposition, name=name, version=version)(*operands, **attrs)
