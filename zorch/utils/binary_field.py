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

from functools import partial
from typing import Any, Literal

import frx
import frx.numpy as fnp
from frx import Array, lax
from frx.experimental import pallas as pl
from frx.experimental.pallas import triton as plgpu

_LIMB = fnp.uint32
_LIMB_BITS = 32
_BIT = fnp.binary_field_t0  # F_2 = GF(2)
_SELECT_BLOCK = 128

BitSelectReduction = Literal["bits", "elements"]


def field_bit_width(dtype: Any) -> int:
    """`W`: the GF(2)-dimension of `dtype` (= its storage bits)."""
    width = fnp.dtype(dtype).itemsize * 8
    if width % _LIMB_BITS != 0:
        raise ValueError(
            f"{fnp.dtype(dtype).name} is {width} bits; the bit kernels work over "
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
    shifts = fnp.arange(_LIMB_BITS, dtype=_LIMB)
    bits = (limbs[..., :, None] >> shifts) & _LIMB(1)
    return bits.reshape(*x.shape, -1)


def _limbs_from_bits(bits: Array) -> Array:
    """`(..., W)` 0/1 `uint32` bits -> `(..., L)` `uint32` limbs: pack each 32
    consecutive bits into a limb, `Σ_r bit_r · 2^r`. The reconstruction inverse of
    [`_bits`]' spread, shared by [`pack`] and the ring-switch tensor-algebra
    transpose. Bits are 0/1 over distinct powers, so the weighted sum equals a
    bit-OR — no overflow."""
    weights = _LIMB(1) << fnp.arange(_LIMB_BITS, dtype=_LIMB)
    return fnp.sum(
        bits.reshape(*bits.shape[:-1], -1, _LIMB_BITS) * weights,
        axis=-1,
        dtype=_LIMB,
    )


def _f2(v: int) -> Array:
    """The F_2 constant `v ∈ {0, 1}` as a `binary_field_t0` scalar. Built via the
    list constructor: a value-cast (`astype`) into `t0` is unlowered and SIGSEGVs,
    and a `uint8 → t0` bitcast is rank-invalid (t0 is 1-bit-logical)."""
    return fnp.asarray(fnp.array([v], _BIT)[0])


def _xor_reduce_rows(x: Array) -> Array:
    """XOR-reduce `x` over axis 0 (length < 256) with eight integer sums.

    Each sum counts one bit position independently in all four bytes of every
    lane; with fewer than 256 rows the byte counters cannot carry into each
    other, so their low bits are exactly the requested parities. `reduce_xor` is
    not lowerable inside a Pallas GPU kernel, so the reduction rides on `sum`.
    """
    result = _LIMB(0)
    byte_lsbs = _LIMB(0x01010101)
    for bit in range(8):
        parity = fnp.sum((x >> bit) & byte_lsbs, axis=0, dtype=_LIMB) & byte_lsbs
        result = result | (parity << bit)
    return result


# The elements reduction sums over all `n` selectors into a tiny `(W, L)` output.
# Gridding on that output gives only `W*L` programs, each serial over `n` — at
# flock m=28 that is 512 programs looping 16384 blocks, ~18x slower than the
# sibling `bits` kernel purely from under-parallelization. Instead grid over `n`:
# each program XOR-folds `iters` sub-blocks of `_ELEMENTS_BLOCK` rows into one
# `(W, L)` accumulator — reading every selector/value once, not `W` times — and
# the per-program partials are XOR-combined. XOR is associative, so the result is
# bit-identical to a serial reduction. `_ELEMENTS_TARGET_PROGRAMS` bounds the
# partials array so the large-`m` shapes #504 cared about stay in memory.
# rows per materialized (BLOCK, W, L) tile; < 256 so _xor_reduce_rows cannot carry
_ELEMENTS_BLOCK = 32
_ELEMENTS_TARGET_PROGRAMS = 8192


def _bit_select_reduce_elements_pallas(
    selectors: Array, values: Array, width: int, limbs: int
) -> Array:
    """Pallas lowering for `(n,) x (n,) -> (W, L)` limb output, parallel over n."""
    n = selectors.shape[0]
    block = _ELEMENTS_BLOCK
    iters = max(1, -(-n // (block * _ELEMENTS_TARGET_PROGRAMS)))
    rows_per_program = block * iters
    programs = -(-n // rows_per_program)
    padded_n = programs * rows_per_program
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))
    values = fnp.pad(values, ((0, padded_n - n), (0, 0)))

    def kernel(selector_ref: Any, value_ref: Any, out_ref: Any) -> None:
        program = pl.program_id(0)
        shifts = fnp.arange(_LIMB_BITS, dtype=_LIMB)

        def body(step: Array, acc: Array) -> Array:
            rows = (
                program * rows_per_program
                + step * block
                + fnp.arange(block, dtype=fnp.int32)
            )
            sel = selector_ref[rows, :]
            val = value_ref[rows, :]
            bits = ((sel[:, :, None] >> shifts) & 1).reshape(block, width)
            return acc ^ _xor_reduce_rows(bits[:, :, None] * val[:, None, :])

        out_ref[program] = lax.fori_loop(
            0, iters, body, fnp.zeros((width, limbs), _LIMB)
        )

    partials = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((programs, width, limbs), _LIMB),
        grid=(programs,),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_elements",
    )(selectors, values)
    return lax.reduce_xor(partials, (0,))


def _bit_select_reduce_bits_pallas(
    selectors: Array, values: Array, width: int, limbs: int
) -> Array:
    """Pallas lowering for `(n,) x (W,) -> (n, L)` limb output."""
    n = selectors.shape[0]
    padded_n = ((n + _SELECT_BLOCK - 1) // _SELECT_BLOCK) * _SELECT_BLOCK
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))

    def kernel(selector_ref: Any, value_ref: Any, out_ref: Any) -> None:
        block = pl.program_id(0)
        limb = pl.program_id(1)
        rows = block * _SELECT_BLOCK + fnp.arange(_SELECT_BLOCK, dtype=fnp.int32)
        acc = fnp.zeros((_SELECT_BLOCK,), dtype=_LIMB)
        for selector_limb in range(limbs):
            packed = selector_ref[rows, selector_limb]
            for shift in range(_LIMB_BITS):
                bit = selector_limb * _LIMB_BITS + shift
                value = value_ref[bit, limb]
                acc ^= fnp.where(((packed >> shift) & 1) != 0, value, 0)
        out_ref[rows, limb] = acc

    out = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((padded_n, limbs), _LIMB),
        grid=(padded_n // _SELECT_BLOCK, limbs),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_bits",
    )(selectors, values)
    return out[:n]


def _bit_select_unpacked_bits_pallas(
    selectors: Array, values: Array, limbs: int
) -> Array:
    """Pallas lowering for 0/1 `(n, B) x (B,) -> (n, L)` limb output."""
    n, bits = selectors.shape
    padded_n = ((n + _SELECT_BLOCK - 1) // _SELECT_BLOCK) * _SELECT_BLOCK
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))

    def kernel(selector_ref: Any, value_ref: Any, out_ref: Any) -> None:
        block = pl.program_id(0)
        limb = pl.program_id(1)
        rows = block * _SELECT_BLOCK + fnp.arange(_SELECT_BLOCK, dtype=fnp.int32)
        acc = fnp.zeros((_SELECT_BLOCK,), dtype=_LIMB)
        for bit in range(bits):
            selected = selector_ref[rows, bit] != 0
            value = value_ref[bit, limb]
            acc ^= fnp.where(selected, value, 0)
        out_ref[rows, limb] = acc

    out = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((padded_n, limbs), _LIMB),
        grid=(padded_n // _SELECT_BLOCK, limbs),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_unpacked_bits",
    )(selectors, values)
    return out[:n]


def _bit_select_packed_bytes_pallas(
    selectors: Array, values: Array, limbs: int
) -> Array:
    """Pallas lowering for packed-byte `(n, B/8) x (B,) -> (n, L)`."""
    n, n_bytes = selectors.shape
    padded_n = ((n + _SELECT_BLOCK - 1) // _SELECT_BLOCK) * _SELECT_BLOCK
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))

    # Flock-core's UniSkipFoldTable: table[j, v] XORs the eight weights selected
    # by byte value v in byte position j. At k_skip=6 this is 8*256*16 = 32 KiB.
    byte_values = values.reshape(n_bytes, 8, limbs)
    byte = fnp.arange(256, dtype=_LIMB)
    shift = fnp.arange(8, dtype=_LIMB)
    selected = ((byte[:, None] >> shift[None, :]) & 1).astype(_LIMB)
    table = lax.reduce_xor(
        selected[None, :, :, None] * byte_values[:, None, :, :], (2,)
    )

    def kernel(selector_ref: Any, table_ref: Any, out_ref: Any) -> None:
        block = pl.program_id(0)
        limb = pl.program_id(1)
        rows = block * _SELECT_BLOCK + fnp.arange(_SELECT_BLOCK, dtype=fnp.int32)
        acc = fnp.zeros((_SELECT_BLOCK,), dtype=_LIMB)
        for byte_index in range(n_bytes):
            value = selector_ref[rows, byte_index].astype(fnp.int32)
            acc ^= table_ref[byte_index, value, limb]
        out_ref[rows, limb] = acc

    out = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((padded_n, limbs), _LIMB),
        grid=(padded_n // _SELECT_BLOCK, limbs),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_packed_bytes",
    )(selectors, table)
    return out[:n]


@frx.jit
def byte_select_xor_reduce(selectors: Array, values: Array) -> Array:
    """XOR weights selected by row-major packed bytes.

    `selectors` is a `uint8` matrix `(n, B/8)`, where byte `j` stores selector
    bits `8*j .. 8*j+7` least-significant-bit first. `values` is a binary-field
    vector `(B,)`; the result `(n,)` XORs `values[b]` wherever row bit `b` is set.

    On GPU a 256-entry XOR table per byte position feeds a Pallas gather, matching
    flock-core's `UniSkipFoldTable`: the kernel reads `B/8` selector bytes per row
    and never expands the `(n, B)` bit matrix. CPU keeps a compact source-level
    expression as the portable oracle.
    """
    if selectors.ndim != 2 or values.ndim != 1:
        raise ValueError("packed-byte bit selection expects 2D selectors and 1D values")
    if fnp.dtype(selectors.dtype) != fnp.dtype(fnp.uint8):
        raise ValueError(
            f"packed-byte selectors must have dtype uint8, got {selectors.dtype}"
        )
    if selectors.shape[0] == 0 or selectors.shape[1] == 0:
        raise ValueError("packed-byte bit selection requires a non-empty matrix")
    bits = selectors.shape[1] * 8
    if values.shape != (bits,):
        raise ValueError(
            f"values must have shape ({bits},) for {selectors.shape[1]} selector "
            f"bytes, got {values.shape}"
        )
    values_dtype = fnp.dtype(values.dtype)
    if not values_dtype.name.startswith("binary_field"):
        raise ValueError("bit-select XOR reduction values must use a binary field")

    limbs = field_bit_width(values.dtype) // _LIMB_BITS
    values_l = _to_limbs(values)
    if frx.default_backend() == "gpu":
        out_l = _bit_select_packed_bytes_pallas(selectors, values_l, limbs)
    else:
        shifts = fnp.arange(8, dtype=fnp.uint8)
        unpacked = ((selectors[:, :, None] >> shifts) & 1).reshape(
            selectors.shape[0], bits
        )
        out_l = lax.reduce_xor(
            unpacked[:, :, None].astype(_LIMB) * values_l[None, :, :], (1,)
        )
    return _from_limbs(out_l, values.dtype)


# Rows per (BLOCK, C, 8, L) tile. 8 keeps the tile at the same register footprint
# as the elements kernel's (32, W, L) despite the extra column axis, and stays
# far below the 256-row carry bound of `_xor_reduce_rows`.
_BYTE_SLICE_BLOCK = 8
_BYTE_SLICE_TARGET_PROGRAMS = 8192


def _byte_slice_reduce_pallas(selectors: Array, values_l: Array, limbs: int) -> Array:
    """Pallas lowering for `(n, C) x (n,) -> (C, 8, L)` limb output, parallel
    over n with per-program partial XOR accumulators (the elements-reduction
    grid of `_bit_select_reduce_elements_pallas`)."""
    n, cols = selectors.shape
    block = _BYTE_SLICE_BLOCK
    iters = max(1, -(-n // (block * _BYTE_SLICE_TARGET_PROGRAMS)))
    rows_per_program = block * iters
    programs = -(-n // rows_per_program)
    padded_n = programs * rows_per_program
    # Triton requires power-of-2 array extents; zero-padded columns contribute
    # all-zero bit slices and are cut off the result.
    padded_cols = 1 << (cols - 1).bit_length() if cols > 1 else 1
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, padded_cols - cols)))
    values_l = fnp.pad(values_l, ((0, padded_n - n), (0, 0)))

    def kernel(selector_ref: Any, value_ref: Any, out_ref: Any) -> None:
        program = pl.program_id(0)
        shifts = fnp.arange(8, dtype=_LIMB)

        def body(step: Array, acc: Array) -> Array:
            rows = (
                program * rows_per_program
                + step * block
                + fnp.arange(block, dtype=fnp.int32)
            )
            sel = selector_ref[rows, :].astype(_LIMB)
            val = value_ref[rows, :]
            bits = (sel[:, :, None] >> shifts) & 1
            return acc ^ _xor_reduce_rows(bits[:, :, :, None] * val[:, None, None, :])

        out_ref[program] = lax.fori_loop(
            0, iters, body, fnp.zeros((padded_cols, 8, limbs), _LIMB)
        )

    partials = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((programs, padded_cols, 8, limbs), _LIMB),
        grid=(programs,),
        compiler_params=plgpu.CompilerParams(),
        name="byte_slice_xor_reduce",
    )(selectors, values_l)
    return lax.reduce_xor(partials, (0,))[:cols]


@frx.jit
def byte_slice_xor_reduce(selectors: Array, values: Array) -> Array:
    """XOR-reduce values into per-(column, bit) slices selected by bytes.

    `selectors` is a `uint8` matrix `(n, C)`; `values` a binary-field vector
    `(n,)`. The result `(C, 8)` XORs `values[i]` into entry `(c, t)` wherever
    bit `t` of `selectors[i, c]` is set — the reduce-over-rows dual of
    `byte_select_xor_reduce` (which combines per row).

    This is the streaming half of an F2-linear table-embedding sum: a weighted
    `Σ_i values[i] · T[selectors[i, c]]` with `T` linear over selector bits
    (`T[v] = ⊕_t bit_t(v)·T[2^t]`) becomes `⊕_t T[2^t] · out[c, t]`, moving the
    per-element field multiply and table gather out of the O(n·C) hot loop into
    an O(8·C) combine. On GPU one pass over selectors and values (grid over n,
    partial XOR accumulators — XOR is associative, so the result is
    bit-identical to a serial reduction); CPU keeps the compact source-level
    expression as the portable oracle.
    """
    if selectors.ndim != 2 or values.ndim != 1:
        raise ValueError("byte-slice reduction expects 2D selectors and 1D values")
    if fnp.dtype(selectors.dtype) != fnp.dtype(fnp.uint8):
        raise ValueError(
            f"byte-slice selectors must have dtype uint8, got {selectors.dtype}"
        )
    n, cols = selectors.shape
    if n == 0 or cols == 0:
        raise ValueError("byte-slice reduction requires a non-empty matrix")
    if values.shape != (n,):
        raise ValueError(
            f"values must have shape ({n},) to match the selector rows, got "
            f"{values.shape}"
        )
    values_dtype = fnp.dtype(values.dtype)
    if not values_dtype.name.startswith("binary_field"):
        raise ValueError("byte-slice reduction values must use a binary field")

    limbs = field_bit_width(values.dtype) // _LIMB_BITS
    values_l = _to_limbs(values)
    if frx.default_backend() == "gpu":
        out_l = _byte_slice_reduce_pallas(selectors, values_l, limbs)
    else:
        shifts = fnp.arange(8, dtype=fnp.uint8)
        bits = ((selectors[:, :, None] >> shifts) & 1).astype(_LIMB)
        out_l = lax.reduce_xor(bits[:, :, :, None] * values_l[:, None, None, :], (0,))
    return _from_limbs(out_l, values.dtype)


@partial(frx.jit, static_argnames="reduce")
def bit_select_xor_reduce(
    selectors: Array,
    values: Array,
    *,
    reduce: BitSelectReduction,
) -> Array:
    """Select binary-field values with packed bits and XOR-reduce one axis.

    `selectors` may be a one-dimensional vector of GF(2^W) elements.
    With `reduce="elements"`, `values` has the same shape and the result has
    shape `(W,)`: result bit-slice `b` XORs `values[i]` wherever bit `b` of
    selector `i` is set. With `reduce="bits"`, `values` has shape `(W,)` and
    the result has the selectors' shape: each result `i` XORs `values[b]`
    wherever bit `b` of selector `i` is set.

    For `reduce="bits"`, selectors may instead be an explicit Boolean/integer
    0/1 matrix of shape `(n, B)` with `values.shape == (B,)`. This is the form
    used when a consumer already holds unpacked witness rows.

    On GPU the Pallas lowering streams directly into the limb output. It never
    materializes the broadcast `(n, W, L)` selection; its working set is one
    fixed-size register block. CPU uses the compact source-level expression so
    the primitive remains portable and easy to validate.
    """
    if values.ndim != 1 or selectors.ndim not in (1, 2):
        raise ValueError(
            "bit-select XOR reduction expects 1D values and 1D packed or 2D "
            "unpacked selectors"
        )
    if selectors.shape[0] == 0:
        raise ValueError("bit-select XOR reduction requires at least one selector")
    values_dtype = fnp.dtype(values.dtype)
    if not values_dtype.name.startswith("binary_field"):
        raise ValueError("bit-select XOR reduction values must use a binary field")

    if selectors.ndim == 2:
        if reduce != "bits":
            raise ValueError('unpacked selectors only support reduce="bits"')
        bits = selectors.shape[1]
        if bits == 0 or values.shape != (bits,):
            raise ValueError(
                f"values must have shape ({bits},) for unpacked selectors, got "
                f"{values.shape}"
            )
        limbs = field_bit_width(values.dtype) // _LIMB_BITS
        values_l = _to_limbs(values)
        if frx.default_backend() == "gpu":
            out_l = _bit_select_unpacked_bits_pallas(selectors, values_l, limbs)
        else:
            selected = selectors.astype(_LIMB)
            out_l = lax.reduce_xor(selected[:, :, None] * values_l[None, :, :], (1,))
        return _from_limbs(out_l, values.dtype)

    if fnp.dtype(selectors.dtype) != values_dtype:
        raise ValueError(
            "bit-select XOR reduction requires selectors and values of the same "
            "binary-field dtype"
        )
    width = field_bit_width(selectors.dtype)
    limbs = width // _LIMB_BITS
    selectors_l = _to_limbs(selectors)
    values_l = _to_limbs(values)

    if reduce == "elements":
        if values.shape != selectors.shape:
            raise ValueError(
                f'values must match selectors for reduce="elements": '
                f"{values.shape} vs {selectors.shape}"
            )
        if frx.default_backend() == "gpu":
            out_l = _bit_select_reduce_elements_pallas(
                selectors_l, values_l, width, limbs
            )
        else:
            bits = _bits(selectors)
            out_l = lax.reduce_xor(bits[:, :, None] * values_l[:, None, :], (0,))
        return _from_limbs(out_l, values.dtype)

    if reduce == "bits":
        if values.shape != (width,):
            raise ValueError(
                f'values must have shape ({width},) for reduce="bits", got '
                f"{values.shape}"
            )
        if frx.default_backend() == "gpu":
            out_l = _bit_select_reduce_bits_pallas(selectors_l, values_l, width, limbs)
        else:
            bits = _bits(selectors)
            out_l = lax.reduce_xor(bits[:, :, None] * values_l[None, :, :], (1,))
        return _from_limbs(out_l, values.dtype)

    raise ValueError(
        f'unknown reduction axis {reduce!r}; expected "bits" or "elements"'
    )


def unpack(x: Array) -> Array:
    """`(...,)` GF(2^W) -> `(..., W)` F_2: the element's F_2-coefficient vector,
    coefficient `r` at index `r`. The GF(2^W) ≅ F_2^W iso, realized as a shift/mask
    over the storage limbs (a bitcast cannot reach sub-byte coefficients) — the
    same bits [`_bits`] returns, retyped from `uint32` to `binary_field_t0`."""
    return fnp.where(_bits(x).astype(bool), _f2(1), _f2(0))


def pack(coeffs: Array, dtype: Any) -> Array:
    """`(..., W)` F_2 -> `(...,)` GF(2^W). Inverse of [`unpack`]: repack each 32
    coefficients into a `uint32` limb (`Σ_r coeff_r · 2^r`), then reinterpret."""
    field_bit_width(dtype)  # validate W is a whole number of limbs
    bit = (coeffs == _f2(1)).astype(_LIMB)
    return _from_limbs(_limbs_from_bits(bit), dtype)
