"""Sponge hash — scheme-agnostic over a Permutation, one entry per construction.

`Sponge.hash(input, sponge_type=...)` absorbs the input in `rate`-sized blocks
and squeezes the first `out` lanes. `sponge_type` (`SpongeType`) selects the
construction — `OVERWRITE` (default; Plonky3 PaddingFreeSponge, the Merkle leaf
hasher) or `CHAINED` (Merkle-Damgard) — as a `_MODES` registry row, so a new
construction adds a member + a row, never a method.

Width comes from `permutation.width`; `rate` and `out` are the free parameters on
`SpongeParams` (capacity = width - rate), like `Poseidon2Params`. A permutation
exposing the fusion seam (`FusedPermutation`) lets this module wrap the whole
absorb as one `zorch.sponge_hash` region the vendor expands into a single
register-resident kernel — the construction is assembled here, over the
permutation's ABI; a bare permutation runs the `while_loop` absorb over its
`permute`. Both read the absorb length at runtime, so a concrete and a symbolic
(shape-poly export) `n` lower the same way — one path, no static-`n` special case.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.fusion import fused_region
from zorch.hash.permutation import FusedPermutation, Permutation


class SpongeType(enum.Enum):
    """Which sponge construction a hash uses. `Sponge.hash(..., sponge_type=...)`
    takes one of these; the behaviour is a `_MODES` registry entry, so a NEW
    construction is a new member + one registry row — no new method on `Sponge`
    and nothing on any permutation (which stays sponge-agnostic)."""

    OVERWRITE = "overwrite"  # Plonky3 PaddingFreeSponge (default)
    CHAINED = "chained"  # Merkle-Damgard (zisk-zorch's linear hash)


# Permutation-agnostic sponge-hash marker: absorb + squeeze as ONE region the
# vendor expands into the fused `sponge_hash` kernel (state register-resident vs
# a per-block permute chain through DRAM). Each permutation family carries its own
# attrs plus a required `permutation` discriminator; the kernel reads the absorb
# length at runtime, so one cubin serves every leaf width and a symbolic width
# exports.
SPONGE_HASH_MARKER = "zorch.sponge_hash"
SPONGE_HASH_MARKER_VERSION = 1


# The sponge constructions differ only in how a block treats its unused (masked)
# rate lanes and its capacity; the loop is otherwise identical, so all ride the
# one permutation-agnostic `_absorb` below via a pair of hooks:
#   `tail_fill(s, rate)`              -> value the masked rate lanes take
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
    SpongeType.OVERWRITE: _Mode(
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
    """Permutation-agnostic absorb as a ``while_loop`` over ``ceil(n/rate)``
    blocks — the shared core for every `SpongeType`. The mode's `tail_fill` /
    `set_capacity` are the only per-construction pieces (see `_MODES`). The loop
    reads its bound at runtime, so it serves concrete and symbolic ``n`` alike;
    that is why the hashes need no static-``n`` path."""
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
    perm: FusedPermutation,
    input: Array,
    rate: int,
    out: int,
    sponge_type: SpongeType,
) -> Array:
    """Absorb + squeeze as ONE `zorch.sponge_hash` region over a permutation that
    exposes the fusion seam — the whole sponge construction, owned here (not on
    the permutation). The decomposition rebuilds a const-free `permute` from the
    region's ABI operands — a `lax.composite` lifts closed-over consts to leading
    operands and would break the emitter ABI — then runs the shared `_absorb`, so
    the region's fallback HLO is byte-identical to the generic path. Only the
    dedicated permutation marks it (the vendor expands the marker into one
    register-resident kernel); a generic one runs the absorb raw so the whole
    sponge stays one LoopFusion, not a loop of per-permute composites.
    """
    operands = perm.fusion_operands(input)

    def sponge(inp: Array, *constants: Array, **_attrs: object) -> Array:
        state = jnp.zeros(perm.width, dtype=inp.dtype)
        return _absorb(
            inp,
            state,
            rate,
            out,
            lambda s: perm.permute_from_operands(s, *constants),
            sponge_type,
        )

    if not perm.has_dedicated_fusion:
        return sponge(*operands)
    # The permutation's identifying attrs plus this sponge's shape: `rate` /
    # `digest_elems` and, for a capacity-chaining construction, the `chained`
    # discriminator the vendor kernel selects on (int — composite bool attrs have
    # no precedent). The marker name/version belong to the sponge, not the
    # permutation, so they live here.
    attrs: dict[str, object] = {
        **perm.fusion_attrs(),
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

    `hash(input, sponge_type=...)` is the one entry point; `sponge_type`
    (`SpongeType`) selects the construction — `OVERWRITE` (default, Plonky3
    padding-free) or `CHAINED` (Merkle-Damgard) — and a new construction is a new
    `SpongeType` + `_MODES` row, not a new method. One call = one function, the
    unit that lowers to one fused `zorch.sponge_hash` kernel. The permutation
    supplies only its arithmetic — `permute`, and (a `FusedPermutation`) its
    fused-region ABI; the sponge construction lives here, not on the permutation.
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

    def hash(
        self, input: Array, sponge_type: SpongeType = SpongeType.OVERWRITE
    ) -> Array:
        """Absorb `input` (1-D) and squeeze the first `out` lanes: (n,) -> (out,).

        `sponge_type` picks the construction (default `OVERWRITE`, the Plonky3
        padding-free sponge; `CHAINED` is the Merkle-Damgard hash — see
        `SpongeType`). A permutation that exposes a dedicated `sponge_hash` (the
        fusion path) absorbs the whole input as one `zorch.sponge_hash` region the
        vendor expands into a single register-resident kernel; a generic
        permutation runs the `while_loop` absorb. Both read the absorb length at
        runtime, so a symbolic `n` (shape-poly export) lowers like a concrete one.
        """
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        # Otherwise a mismatch surfaces deep in the absorb (input mixed with the
        # round constants) as an opaque promotion error; also gates EF rows out.
        if input.dtype != self.dtype:
            raise TypeError(
                f"input dtype {input.dtype} must match the sponge field {self.dtype}"
            )
        # A capacity-filling construction (e.g. CHAINED) carries the out-lane
        # digest as the whole capacity, so it needs rate + out == width.
        if _MODES[sponge_type].fills_capacity and (
            self.rate + self.out != self._permutation.width
        ):
            raise ValueError(
                f"{sponge_type.value} hash needs rate + out == width (the digest "
                f"fills the capacity), got {self.rate} + {self.out} != "
                f"{self._permutation.width}"
            )
        # A permutation exposing the fusion seam lowers the whole sponge as one
        # `zorch.sponge_hash` region (built here — the construction is the
        # sponge's, the ABI the permutation's); a bare permutation runs the
        # generic `while_loop` absorb over its `permute`.
        perm = self._permutation
        if isinstance(perm, FusedPermutation):
            return _fused_hash(perm, input, self.rate, self.out, sponge_type)
        state = jnp.zeros(perm.width, dtype=input.dtype)
        return _absorb(input, state, self.rate, self.out, perm.permute, sponge_type)
