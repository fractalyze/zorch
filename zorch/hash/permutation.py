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

    # -- Fused-region ABI. A consumer (e.g. `Sponge`) wraps a whole computation
    # built from this permutation — an `absorb`/`squeeze`, a compression — as ONE
    # `fused_region` in this permutation's ABI without knowing the operand layout:
    # the consumer owns the region's marker name and its own attributes, the
    # permutation owns only its arithmetic. These are exercised only on the
    # dedicated path (`has_dedicated_fusion`); a non-fused permutation supplies
    # inert conformant stubs (consumers iterate `permute` for it). The invariant:
    # ``permute_from_operands(state, *fusion_operands(x)[1:])`` runs the
    # permutation on ``state`` (marker-free), and ``fusion_operands(x)[0] is x``.

    def fusion_operands(self, leading: Array) -> tuple[Array, ...]:
        """The composite operands ``(leading, *constants)`` — the region's ABI.
        ``leading`` is the caller's data (a sponge input, a state); the rest are
        this permutation's constants (round constants), passed explicitly so a
        `lax.composite` does not lift them to leading operands and break the ABI.
        """
        ...

    def permute_from_operands(self, state: Array, *operands: Array) -> Array:
        """Run the permutation on ``state`` given the constant operands (the tail
        of `fusion_operands`), as a straight-line body with no captured consts —
        the decomposition a `fused_region` runs."""
        ...

    def fusion_attrs(self) -> dict[str, Any]:
        """This permutation's identifying `composite.attributes` (a `permutation`
        discriminator plus the shape the vendor recognizer reads). Meaningful only
        on the dedicated path (`has_dedicated_fusion`)."""
        ...
