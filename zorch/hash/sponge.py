"""Padding-free, overwrite-mode sponge hash — scheme-agnostic over a Permutation.

Absorb the input in `rate`-sized blocks, overwriting the first `rate` lanes of
the state (replace, not XOR) and permuting after each; no padding is added, so a
final partial block overwrites only its own lanes. Squeeze the first `out` lanes.
This is the Merkle leaf hasher (Plonky3 PaddingFreeSponge).

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. The first
block (and a partial last block) absorb outside the loop; the remaining full
blocks run in one `lax.scan`, keeping trace and lowering cost constant in the
input width instead of paying one permute trace per block (#135).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.hash.permutation import Permutation


def _absorb_symbolic(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
) -> Array:
    """Shape-polymorphic absorb for a symbolic ``n`` (export only): a ``while_loop``
    over ``ceil(n/rate)`` blocks, byte-identical to the concrete ``Sponge.hash``.
    Used where ``n // rate`` is undecidable; shared by ``Sponge.hash`` and the
    poseidon2 sponge-hash marker body so the two cannot drift."""
    n = input.shape[0]
    nb = (n + rate - 1) // rate
    lanes = jnp.arange(rate)

    def cond(carry: tuple[Array, Array]) -> Array:
        return carry[1] < nb

    def body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        s, i = carry
        start = i * rate
        w = jnp.minimum(rate, n - start)
        # Last block reads past n; clamp OOB indices (masked out below).
        block = input[jnp.clip(start + lanes, 0, n - 1)]
        s = s.at[:rate].set(jnp.where(lanes < w, block, s[:rate]))
        return permute(s), i + 1

    state, _ = lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state[:out]


@dataclass(frozen=True)
class SpongeParams:
    """Free parameters of a padding-free sponge.

    rate : field elements absorbed per permutation (capacity = width - rate).
    out  : field elements squeezed (the digest size).

    Contract: rate >= 1 and out >= 1 (validated here); rate < permutation.width
    and out <= permutation.width (validated by ``Sponge``, which knows the width).
    """

    rate: int
    out: int

    def __post_init__(self) -> None:
        if self.rate < 1:
            raise ValueError(f"rate ({self.rate}) must be >= 1")
        if self.out < 1:
            raise ValueError(f"out ({self.out}) must be >= 1")


class Sponge:
    """Padding-free, overwrite-mode sponge over a fixed-width Permutation.

    hash = overwrite state[:rate] with each input block -> permute (repeat) ->
    first `out` lanes. One call is one function — the unit that lowers to one
    fused kernel.
    """

    def __init__(self, permutation: Permutation, params: SpongeParams) -> None:
        if params.rate >= permutation.width:
            raise ValueError(
                f"rate ({params.rate}) must be < permutation width "
                f"({permutation.width})"
            )
        if params.out > permutation.width:
            raise ValueError(
                f"out ({params.out}) must be <= permutation width "
                f"({permutation.width})"
            )
        self._permutation = permutation
        self.rate = params.rate
        self.out = params.out

    # Value equality/hash for static jit-zone keys, like the Permutation
    # contract it builds on (#214) — identity equality re-traces per instance.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sponge):
            return NotImplemented
        return (self._permutation, self.rate, self.out) == (
            other._permutation,
            other.rate,
            other.out,
        )

    def __hash__(self) -> int:
        return hash((self._permutation, self.rate, self.out))

    @property
    def has_dedicated_fusion(self) -> bool:
        """Whether the permutation lowers to a hash-dedicated fusion marker, so a
        consumer can wrap a whole region using this hash (e.g. a Merkle commit) in
        an expandable composite. Delegates to the permutation; names no hash."""
        return self._permutation.has_dedicated_fusion

    @property
    def dtype(self) -> Any:
        """The base field the sponge absorbs and squeezes (the permutation's)."""
        return self._permutation.dtype

    def hash(self, input: Array) -> Array:
        """Absorb `input` (1-D) and squeeze: (n,) over dtype -> (out,)."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        state = jnp.zeros(self._permutation.width, dtype=input.dtype)
        n = input.shape[0]
        if not isinstance(n, int):  # shape-poly export: symbolic n
            # Dedicated permutation → fused sponge kernel (reads the absorb length
            # at runtime); generic → the inline while_loop absorb.
            fused = getattr(self._permutation, "sponge_hash", None)
            if fused is not None:
                return fused(input, self.rate, self.out)
            return _absorb_symbolic(
                input, state, self.rate, self.out, self._permutation.permute
            )
        if n == 0:
            return state[: self.out]
        if n <= self.rate:  # single (possibly partial) block
            state = state.at[:n].set(input)
            return self._permutation.permute(state)[: self.out]
        # Block 0 absorbs outside the scan so the first permute is a top-level
        # marker (not nested in a loop body); the rest fold in one scan below.
        state = state.at[: self.rate].set(input[: self.rate])
        state = self._permutation.permute(state)
        # Remaining full blocks in one scan: trace cost constant in n (#135).
        full = n // self.rate
        if full > 1:
            blocks = input[self.rate : full * self.rate].reshape(full - 1, self.rate)

            def absorb(s: Array, block: Array) -> tuple[Array, None]:
                s = s.at[: self.rate].set(block)
                return self._permutation.permute(s), None

            state, _ = lax.scan(absorb, state, blocks)
        # A partial last block overwrites only its own lanes (padding-free).
        tail = n - full * self.rate
        if tail:
            state = state.at[:tail].set(input[full * self.rate :])
            state = self._permutation.permute(state)
        return state[: self.out]
