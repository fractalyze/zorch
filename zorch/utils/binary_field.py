# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Binary-field ⟷ F_2 bit-vector representation for any binary tower dtype
(`binary_field_ghash`, `binary_field_t*`).

An element of GF(2^W) is a W-dimensional vector over F_2; [`unpack`] exposes that
coefficient vector and [`pack`] rebuilds the element — the GF(2^W) ≅ F_2^W
isomorphism the ring-switch / tensor-algebra kernels ride on.

It is a shift/mask, NOT a bitcast. A hardware bitcast is byte-granular (the finest
binary-field bitcast is GF(2^128) ↔ GF(2^8); GF(2^128) → F_2 fails the
compatible-width check), so reaching the individual F_2 coefficients needs the
explicit unpack. The `uint32` storage limbs the shift/mask rides on stay private.
`uint32` is the finest limb, so it keeps the bit kernels valid for the widest set
of tower dtypes — [`field_bit_width`] needs the width to be a whole number of
limbs, so a `uint64` limb would reject the 32-bit tower level (`binary_field_t5`)
— and keeps the ring-switch `{0, 1} × limb` products narrow.
"""
from __future__ import annotations

from typing import Any

import frx.numpy as jnp
from frx import Array, lax

_LIMB = jnp.uint32
_LIMB_BITS = 32
_BIT = jnp.binary_field_t0  # F_2 = GF(2)


def field_bit_width(dtype: Any) -> int:
    """`W`: the GF(2)-dimension of `dtype` (= its storage bits)."""
    width = jnp.dtype(dtype).itemsize * 8
    if width % _LIMB_BITS != 0:
        raise ValueError(
            f"{jnp.dtype(dtype).name} is {width} bits; the bit kernels work over "
            f"uint{_LIMB_BITS} limbs and need a multiple of {_LIMB_BITS}"
        )
    return width


def _to_limbs(x: Array) -> Array:
    """`(...,)` binary field -> `(..., L)` little-endian `uint32` storage limbs.

    Private: the storage granularity the F_2 shift/mask rides on, not a field
    concept. Public code uses [`unpack`] / [`pack`]."""
    return lax.bitcast_convert_type(x, _LIMB)


def _from_limbs(limbs: Array, dtype: Any) -> Array:
    """`(..., L)` `uint32` limbs -> `(...,)` binary field. Inverse of [`_to_limbs`]."""
    return lax.bitcast_convert_type(limbs, dtype)


def _bits(x: Array) -> Array:
    """`(...,)` binary field -> `(..., W)` 0/1 `uint32`: bit `r` is limb `r // 32`'s
    bit `r % 32`.

    Private: the `uint32`-limb bit form the ring-switch bit-decomposition kernels
    ride on — they multiply `{0, 1} × uint32 limb`, reinterpret the product to the
    field and sum it, which needs the bits *as integers*, not the F_2
    (`binary_field_t0`) view [`unpack`] hands public callers."""
    limbs = _to_limbs(x)
    shifts = jnp.arange(_LIMB_BITS, dtype=_LIMB)
    bits = (limbs[..., :, None] >> shifts) & _LIMB(1)
    return bits.reshape(*x.shape, -1)


def _limbs_from_bits(bits: Array) -> Array:
    """`(..., W)` 0/1 `uint32` bits -> `(..., L)` `uint32` limbs: pack each 32
    consecutive bits into a limb, `Σ_r bit_r · 2^r`. The reconstruction inverse of
    [`_bits`]' spread, shared by [`pack`] and the ring-switch tensor-algebra
    transpose. Bits are 0/1 over distinct powers, so the weighted sum equals a
    bit-OR — no overflow."""
    weights = _LIMB(1) << jnp.arange(_LIMB_BITS, dtype=_LIMB)
    return jnp.sum(
        bits.reshape(*bits.shape[:-1], -1, _LIMB_BITS) * weights,
        axis=-1,
        dtype=_LIMB,
    )


def _f2(v: int) -> Array:
    """The F_2 constant `v ∈ {0, 1}` as a `binary_field_t0` scalar. Built via the
    list constructor: a value-cast (`astype`) into `t0` is unlowered and SIGSEGVs,
    and a `uint8 → t0` bitcast is rank-invalid (t0 is 1-bit-logical)."""
    return jnp.asarray(jnp.array([v], _BIT)[0])


def unpack(x: Array) -> Array:
    """`(...,)` GF(2^W) -> `(..., W)` F_2: the element's F_2-coefficient vector,
    coefficient `r` at index `r`. The GF(2^W) ≅ F_2^W iso, realized as a shift/mask
    over the storage limbs (a bitcast cannot reach sub-byte coefficients) — the
    same bits [`_bits`] returns, retyped from `uint32` to `binary_field_t0`."""
    return jnp.where(_bits(x).astype(bool), _f2(1), _f2(0))


def pack(coeffs: Array, dtype: Any) -> Array:
    """`(..., W)` F_2 -> `(...,)` GF(2^W). Inverse of [`unpack`]: repack each 32
    coefficients into a `uint32` limb (`Σ_r coeff_r · 2^r`), then reinterpret."""
    field_bit_width(dtype)  # validate W is a whole number of limbs
    bit = (coeffs == _f2(1)).astype(_LIMB)
    return _from_limbs(_limbs_from_bits(bit), dtype)
