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
        """
        ...
