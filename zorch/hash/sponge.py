"""Sponge hash — scheme-agnostic over a Permutation, one entry per construction.

`Sponge.hash(input, sponge_type=...)` absorbs in `rate`-sized blocks and squeezes
the first `out` lanes. `sponge_type` (`SpongeType`) picks the construction via a
`_MODES` row, so a new one is a member + a row, never a method.

`rate`/`out` are the free params on `SpongeParams` (capacity = width - rate). A
`has_dedicated_fusion` permutation lowers the whole absorb to one
`zorch.sponge_hash` region (assembled here over the permutation's fused-region
ABI); a non-fused one runs the `while_loop` absorb over `permute`. Both read the
absorb length at runtime, so concrete and symbolic `n` lower the same way.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.fusion import fused_region
from zorch.hash.permutation import Permutation


class SpongeType(enum.Enum):
    """The sponge construction a hash uses (`Sponge.hash(..., sponge_type=...)`).
    Behaviour is a `_MODES` row, so a new one is a member + a row, not a method."""

    PADDING_FREE = "padding_free"  # Plonky3 PaddingFreeSponge (default)
    CHAINED = "chained"  # the Merkle-Damgard construction


# The whole absorb+squeeze as one region the vendor expands into the fused kernel.
# Attrs carry a `permutation` discriminator + shape; the kernel reads the absorb
# length at runtime, so one cubin serves every width and symbolic `n`.
SPONGE_HASH_MARKER = "zorch.sponge_hash"
SPONGE_HASH_MARKER_VERSION = 1


# Constructions differ only in how a block fills its masked rate lanes and its
# capacity; the loop is otherwise shared (`_absorb`), via two hooks. `cap` is the
# block's prior digest (`s[:out]`). Zeros are const-free (`s[:r]-s[:r]`, not
# `jnp.zeros`) so no fresh const is lifted to a leading operand, breaking the ABI.
def _keep_prior(s: Array, rate: int) -> Array:
    """Padding-free: a masked lane keeps the prior state."""
    return s[:rate]


def _zero_pad(s: Array, rate: int) -> Array:
    """Chained: a masked lane is zero-padded (const-free zero)."""
    return s[:rate] - s[:rate]


def _carry_capacity(s: Array, cap: Array, rate: int, out: int) -> Array:
    """Padding-free: capacity carries implicitly."""
    return s


def _chain_digest(s: Array, cap: Array, rate: int, out: int) -> Array:
    """Chained: capacity lanes [rate:rate+out] take the prior digest."""
    return s.at[rate : rate + out].set(cap)


@dataclass(frozen=True)
class _Mode:
    """One sponge construction's behaviour — the whole per-type surface. Add a
    `SpongeType` member and a `_MODES` row to introduce a new construction."""

    tail_fill: Callable[[Array, int], Array]
    set_capacity: Callable[[Array, Array, int, int], Array]
    # The digest fills the whole capacity, so the caller must have
    # rate + out == width (chaining carries the out-lane digest as capacity).
    fills_capacity: bool
    # `chained` marker attribute the vendor kernel reads (int; absent when 0).
    marker_chained: int


_MODES: dict[SpongeType, _Mode] = {
    SpongeType.PADDING_FREE: _Mode(
        tail_fill=_keep_prior,
        set_capacity=_carry_capacity,
        fills_capacity=False,
        marker_chained=0,
    ),
    SpongeType.CHAINED: _Mode(
        tail_fill=_zero_pad,
        set_capacity=_chain_digest,
        fills_capacity=True,
        marker_chained=1,
    ),
}


def _absorb(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
    sponge_type: SpongeType,
) -> Array:
    """Absorb as a ``while_loop`` over ``ceil(n/rate)`` blocks — shared by every
    `SpongeType` (the mode's `tail_fill`/`set_capacity` are the only per-type
    pieces). The bound is read at runtime, so concrete and symbolic ``n`` alike."""
    mode = _MODES[sponge_type]
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
        s = s.at[:rate].set(jnp.where(lanes < w, block, mode.tail_fill(s, rate)))
        s = mode.set_capacity(s, cap, rate, out)
        return permute(s), i + 1

    state, _ = lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state[:out]


def _fused_hash(
    perm: Permutation,
    input: Array,
    rate: int,
    out: int,
    sponge_type: SpongeType,
) -> Array:
    """Absorb+squeeze as ONE `zorch.sponge_hash` region over a dedicated-fusion
    permutation. The decomposition rebuilds a const-free `permute` from the ABI
    operands (a `lax.composite` lifts closed-over consts and breaks the ABI), then
    runs `_absorb`, so the fallback HLO matches the generic path. Caller gates on
    `has_dedicated_fusion`."""
    operands, permute_from_operands, perm_attrs = perm.fused_region_spec(input)

    def sponge(inp: Array, *constants: Array, **_attrs: object) -> Array:
        state = jnp.zeros(perm.width, dtype=inp.dtype)
        return _absorb(
            inp,
            state,
            rate,
            out,
            lambda s: permute_from_operands(s, *constants),
            sponge_type,
        )

    # Permutation attrs + this sponge's shape (`rate`/`digest_elems`, and the
    # `chained` discriminator the kernel selects on — int, since composite bool
    # attrs have no precedent). The marker name/version are the sponge's.
    attrs: dict[str, object] = {
        **perm_attrs,
        "rate": rate,
        "digest_elems": out,
    }
    marker_chained = _MODES[sponge_type].marker_chained
    if marker_chained:
        attrs["chained"] = marker_chained
    return fused_region(
        sponge,
        *operands,
        name=SPONGE_HASH_MARKER,
        version=SPONGE_HASH_MARKER_VERSION,
        **attrs,
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
    """Sponge hash over a fixed-width Permutation.

    `hash(input, sponge_type=...)` is the one entry point (one call = one function
    = one fused `zorch.sponge_hash` kernel). The permutation supplies only its
    arithmetic — `permute` and its fused-region ABI; the construction lives here.
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
        consumer can wrap a whole region using this hash in an expandable
        composite. Delegates to the permutation."""
        return self._permutation.has_dedicated_fusion

    @property
    def dtype(self) -> Any:
        """The base field the sponge absorbs and squeezes (the permutation's)."""
        return self._permutation.dtype

    def hash(
        self, input: Array, sponge_type: SpongeType = SpongeType.PADDING_FREE
    ) -> Array:
        """Absorb `input` (1-D) and squeeze the first `out` lanes: (n,) -> (out,).

        `sponge_type` picks the construction (see `SpongeType`). A
        `has_dedicated_fusion` permutation lowers to one `zorch.sponge_hash`
        region; a non-fused one runs the `while_loop` absorb. Both read the absorb
        length at runtime, so symbolic `n` lowers like a concrete one.
        """
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        # Else a mismatch surfaces deep in the absorb as an opaque promotion error.
        if input.dtype != self.dtype:
            raise TypeError(
                f"input dtype {input.dtype} must match the sponge field {self.dtype}"
            )
        # A capacity-filling construction (e.g. CHAINED) needs rate + out == width.
        if _MODES[sponge_type].fills_capacity and (
            self.rate + self.out != self._permutation.width
        ):
            raise ValueError(
                f"{sponge_type.value} hash needs rate + out == width (the digest "
                f"fills the capacity), got {self.rate} + {self.out} != "
                f"{self._permutation.width}"
            )
        # Dedicated-fusion → one `zorch.sponge_hash` region (built here); else the
        # generic `while_loop` absorb over `permute`.
        perm = self._permutation
        if perm.has_dedicated_fusion:
            return _fused_hash(perm, input, self.rate, self.out, sponge_type)
        state = jnp.zeros(perm.width, dtype=input.dtype)
        return _absorb(input, state, self.rate, self.out, perm.permute, sponge_type)
