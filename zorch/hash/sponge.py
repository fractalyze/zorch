"""Padding-free, overwrite-mode sponge hash — scheme-agnostic over a Permutation.

Absorb the input in `rate`-sized blocks, overwriting the first `rate` lanes of
the state (replace, not XOR) and permuting after each; no padding is added, so a
final partial block overwrites only its own lanes. Squeeze the first `out` lanes.
This is the Merkle leaf hasher (Plonky3 PaddingFreeSponge).

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. The first
block (and a partial last block) absorb outside the loop, so the traced body
always carries a top-level permute — the marker a `zorch.merkle_commit`
consumer reads the hash identity from; the remaining full blocks run in one
`lax.scan`, keeping trace and lowering cost constant in the input width
instead of paying one permute trace per block (#135).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from zorch.hash.permutation import Permutation


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

    def hash(self, input: Array) -> Array:
        """Absorb `input` (1-D) and squeeze: (n,) over dtype -> (out,)."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        state = jnp.zeros(self._permutation.width, dtype=input.dtype)
        n = input.shape[0]
        if n == 0:
            return state[: self.out]
        if n <= self.rate:  # single (possibly partial) block
            state = state.at[:n].set(input)
            return self._permutation.permute(state)[: self.out]
        # Block 0 MUST absorb outside the scan: a merkle_commit consumer
        # discovers the hash identity from a top-level permute marker only
        # (it does not look inside loop bodies), so folding block 0 into the
        # scan silently breaks commit. Revisit if marker discovery learns to
        # recurse.
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

    def hash_batched(self, inputs: Array) -> Array:
        """Batch of `hash`: (b, n) -> (b, out), numerically a `vmap(hash)`.

        Absorbs the whole Merkle leaf level at once on `(b, width)` states,
        routing every permute through `permute_batched`. With a dedicated-fusion
        permutation that makes the ragged Merkle layers share one lowered permute
        body instead of re-emitting it per layer. All `b` leaves
        share the input length `n`, so the block structure is static, exactly as in
        `hash` — including absorbing block 0 outside the scan so the top-level
        permute marker stays discoverable to a `merkle_commit` consumer."""
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be 2-D, got ndim={inputs.ndim}")
        b, n = inputs.shape
        state = jnp.zeros((b, self._permutation.width), dtype=inputs.dtype)
        if n == 0:
            return state[:, : self.out]
        if n <= self.rate:  # single (possibly partial) block
            state = state.at[:, :n].set(inputs)
            return self._permutation.permute_batched(state)[:, : self.out]
        # Block 0 outside the scan (see `hash`): a merkle_commit consumer reads
        # the hash identity from a top-level permute marker, not inside the loop.
        state = state.at[:, : self.rate].set(inputs[:, : self.rate])
        state = self._permutation.permute_batched(state)
        full = n // self.rate
        if full > 1:
            blocks = inputs[:, self.rate : full * self.rate].reshape(
                b, full - 1, self.rate
            )
            # Scan over the block axis (one permute body for all blocks, #135);
            # move it leading so each step carries a `(b, rate)` block.
            blocks = jnp.swapaxes(blocks, 0, 1)

            def absorb(s: Array, block: Array) -> tuple[Array, None]:
                s = s.at[:, : self.rate].set(block)
                return self._permutation.permute_batched(s), None

            state, _ = lax.scan(absorb, state, blocks)
        # A partial last block overwrites only its own lanes (padding-free).
        tail = n - full * self.rate
        if tail:
            state = state.at[:, :tail].set(inputs[:, full * self.rate :])
            state = self._permutation.permute_batched(state)
        return state[:, : self.out]
