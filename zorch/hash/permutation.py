"""The Permutation seam every symmetric primitive builds on.

A fixed-width permutation over a single field dtype. Consumers (duplex sponge,
Fiat-Shamir transcript, Merkle compression) read `width` to size state and
`dtype` to allocate it, then call `permute` — they never name a concrete hash.
Poseidon2 is one implementation; any other fixed-width permutation drops in
unchanged.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a permutation rides pytree aux (`DuplexTranscript`
meta_fields), where identity equality silently re-traces the enclosing jit
zone on every freshly built instance. A Protocol cannot enforce
this — each implementation carries it (`Poseidon2`, `CheapPermutation`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from frx import Array


@runtime_checkable
class Permutation(Protocol):
    width: int  # state length (rate + capacity)
    dtype: Any  # field dtype of each state element
    # Whether `permute` lowers to a hash-dedicated fusion marker (vs the generic
    # region marker). When true, a vendor can expand a whole-region composite —
    # e.g. a Merkle commit — by reading this hash's marker; consumers gate that
    # wrapping on it without naming a concrete hash.
    has_dedicated_fusion: bool
    # The composite name + version `permute`'s marker carries — what a consumer
    # needs to RE-MARK a permute inside its own composite decomposition (a duplex
    # absorb chain is one), so the inlined fallback keeps the dedicated
    # per-permute kernels instead of raw permute bodies.
    #
    # Name and version travel together because they are one ABI coordinate: a
    # contract change stages through `composite.version` rather than a rename (see
    # `zorch.fusion`), so a consumer holding the name alone can re-mark against a
    # stale contract. An undedicated permutation reports the generic marker at
    # version 0, which is what `has_dedicated_fusion` is read off.
    fused_region_marker: tuple[str, int]

    def permute(self, state: Array) -> Array:
        """Apply the permutation: (width,) over `dtype` -> (width,).

        One call is one function — the unit that lowers to one fused kernel.
        Batch with `frx.vmap(permute)`: the dedicated-fusion marker lowers
        identically batched (one shared decomposition), so no batched twin is
        needed.
        """
        ...

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """Pieces to wrap a computation over this permutation as ONE `fused_region`
        in its ABI, without the consumer knowing the operand layout. Returns
        ``(operands, permute_from_operands, attrs)``:

        - ``operands`` = ``(leading, *constants)``; the round constants ride as
          explicit operands so a `lax.composite` can't lift them and break the ABI.
        - ``permute_from_operands(state, *constants)`` = the const-free permute the
          `fused_region` runs.
        - ``attrs`` = identifying `composite.attributes` (a ``permutation``
          discriminator + shape).

        Meaningful only on the dedicated path; a non-fused permutation returns an
        inert spec.
        """
        ...
