"""Padding-free, overwrite-mode sponge hash — scheme-agnostic over a Permutation.

Absorb the input in `rate`-sized blocks, overwriting the first `rate` lanes of
the state (replace, not XOR) and permuting after each; no padding is added, so a
final partial block overwrites only its own lanes. Squeeze the first `out` lanes.
This is the Merkle leaf hasher (Plonky3 PaddingFreeSponge).

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. A permutation
exposing a dedicated `sponge_hash` (the poseidon2 fusion path) absorbs the whole
input as one `zorch.sponge_hash` region the vendor expands into a single
register-resident kernel; a generic permutation runs the `while_loop` absorb. Both
read the absorb length at runtime, so a concrete and a symbolic (shape-poly
export) `n` lower the same way — one path, no static-`n` special case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.hash.permutation import Permutation

# Permutation-agnostic sponge-hash marker: absorb + squeeze as ONE region the
# vendor expands into the fused `sponge_hash` kernel (state register-resident vs
# a per-block permute chain through DRAM). Each permutation family carries its own
# attrs plus a required `permutation` discriminator; the kernel reads the absorb
# length at runtime, so one cubin serves every leaf width and a symbolic width
# exports.
SPONGE_HASH_MARKER = "zorch.sponge_hash"
SPONGE_HASH_MARKER_VERSION = 1


# The two absorb constructions differ only in how a block treats its unused
# (masked) rate lanes and its capacity; the loop is otherwise identical, so both
# ride the one permutation-agnostic `_absorb` below via a pair of hooks:
#   `tail_fill(s, rate)`         -> value the masked rate lanes take
#   `set_capacity(s, cap, rate, out)` -> state after writing the capacity lanes
# `cap` is the block's PRIOR digest (`s[:out]`), snapshot before the overwrite.
# Kept const-free (zeros via `s[:r] - s[:r]`, not `jnp.zeros`) so the values stay
# derived from the carry — a fresh const would be lifted to a leading operand and
# break the fused marker's operand ABI.
def _keep_prior(s: Array, rate: int) -> Array:
    """Overwrite mode: a masked lane keeps the prior state (no padding)."""
    return s[:rate]


def _zero_pad(s: Array, rate: int) -> Array:
    """Chained mode: a masked lane is zero-padded (const-free zero)."""
    return s[:rate] - s[:rate]


def _carry_capacity(s: Array, cap: Array, rate: int, out: int) -> Array:
    """Overwrite mode: capacity is the prior state's, carried implicitly."""
    return s


def _chain_digest(s: Array, cap: Array, rate: int, out: int) -> Array:
    """Chained mode: capacity lanes [rate:rate+out] take the prior digest."""
    return s.at[rate : rate + out].set(cap)


def _absorb(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
    *,
    tail_fill: Callable[[Array, int], Array],
    set_capacity: Callable[[Array, Array, int, int], Array],
) -> Array:
    """Permutation-agnostic absorb as a ``while_loop`` over ``ceil(n/rate)``
    blocks — the shared core of both `_absorb_overwrite` and `_absorb_chained`.

    The loop reads its bound at runtime, so it serves concrete and symbolic ``n``
    alike; that is why the hashes need no static-``n`` path. `tail_fill` and
    `set_capacity` are the only per-construction pieces (see above)."""
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
        cap = s[:out]  # prior digest (zeros on block 0); snapshot before overwrite
        s = s.at[:rate].set(jnp.where(lanes < w, block, tail_fill(s, rate)))
        s = set_capacity(s, cap, rate, out)
        return permute(s), i + 1

    state, _ = lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state[:out]


def _absorb_overwrite(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
) -> Array:
    """Padding-free overwrite sponge (Plonky3 PaddingFreeSponge): a short final
    block overwrites only its own lanes, and the capacity carries the prior state
    forward. The fallback for a permutation with no dedicated sponge kernel, and
    the body of the `sponge_hash` marker — shared so the two cannot drift."""
    return _absorb(
        input, state, rate, out, permute,
        tail_fill=_keep_prior, set_capacity=_carry_capacity,
    )


def _absorb_chained(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
) -> Array:
    """Chained (Merkle-Damgard) hash: a short final block is zero-padded, and each
    block carries the prior block's digest (``s[:out]``) in the capacity lanes
    ``[rate:rate+out]``. The digest fills the whole capacity, so requires
    ``rate + out == width`` (validated by the caller)."""
    return _absorb(
        input, state, rate, out, permute,
        tail_fill=_zero_pad, set_capacity=_chain_digest,
    )


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
        """Absorb `input` (1-D) and squeeze: (n,) over dtype -> (out,).

        A permutation that exposes a dedicated `sponge_hash` (the poseidon2
        fusion path) absorbs the whole input as one `zorch.sponge_hash`
        region the vendor expands into a single register-resident kernel; a
        generic permutation runs the `while_loop` absorb. Both read the absorb
        length at runtime, so a symbolic `n` (shape-poly export) lowers exactly
        like a concrete one — one path, no static-`n` special case.
        """
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        # Otherwise a mismatch surfaces deep in the absorb (input mixed with the
        # round constants) as an opaque promotion error; also gates EF rows out.
        if input.dtype != self.dtype:
            raise TypeError(
                f"input dtype {input.dtype} must match the sponge field {self.dtype}"
            )
        fused = getattr(self._permutation, "overwrite_hash", None)
        if fused is not None:
            return fused(input, self.rate, self.out)
        state = jnp.zeros(self._permutation.width, dtype=input.dtype)
        return _absorb_overwrite(
            input, state, self.rate, self.out, self._permutation.permute
        )
