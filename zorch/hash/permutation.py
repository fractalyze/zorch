"""The Permutation seam every symmetric primitive builds on.

A fixed-width permutation over a single field dtype. Consumers (duplex sponge,
Fiat-Shamir transcript, Merkle compression) read `width` to size state and
`dtype` to allocate it, then call `permute` — they never name a concrete hash.
Poseidon2 is one implementation; any other fixed-width permutation drops in
unchanged.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a permutation rides pytree aux (`DuplexTranscript`
meta_fields), where identity equality silently re-traces the enclosing jit
zone on every freshly built instance (issue #163). A Protocol cannot enforce
this — each implementation carries it (`Poseidon2`, `CheapPermutation`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Permutation(Protocol):
    width: int  # state length (rate + capacity)
    dtype: Any  # field dtype of each state element
    # Whether `permute` lowers to a hash-dedicated fusion marker (vs the generic
    # region marker). When true, a vendor can expand a whole-region composite —
    # e.g. a Merkle commit — by reading this hash's marker; consumers gate that
    # wrapping on it without naming a concrete hash.
    has_dedicated_fusion: bool

    def permute(self, state: Array) -> Array:
        """Apply the permutation: (width,) over `dtype` -> (width,).

        One call is one function — the unit that lowers to one fused kernel.
        Batch with `jax.vmap(permute)`: the dedicated-fusion marker lowers
        identically batched (one shared decomposition), so no batched twin is
        needed.
        """
        ...

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The pieces to wrap a whole computation over this permutation — an
        `absorb`/`squeeze`, a compression — as ONE `fused_region` in this
        permutation's ABI, without the consumer (e.g. `Sponge`) knowing the
        operand layout. Returns ``(operands, permute_from_operands, attrs)``:

        - ``operands`` = the composite operands ``(leading, *constants)``;
          ``leading`` is the caller's data, the rest this permutation's round
          constants, passed explicitly so a `lax.composite` cannot lift them to
          leading operands and break the emitter ABI.
        - ``permute_from_operands(state, *constants)`` runs the permutation on
          ``state`` from those constant operands — a const-free straight-line
          body, the decomposition the `fused_region` runs.
        - ``attrs`` = identifying `composite.attributes` (a ``permutation``
          discriminator plus the shape the recognizer reads); the consumer owns
          the marker name and its own attributes.

        Invariant: ``operands[0] is leading`` and
        ``permute_from_operands(s, *operands[1:])`` runs the permutation on ``s``
        marker-free. Meaningful only on the dedicated path
        (`has_dedicated_fusion`); a non-fused permutation returns an inert spec.
        """
        ...
