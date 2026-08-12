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
import frx.ffi
import frx.numpy as fnp
from frx import Array, lax
from frx.experimental import pallas as pl
from frx.experimental.pallas import triton as plgpu

_LIMB = fnp.uint32
_LIMB_BITS = 32
_BIT = fnp.binary_field_t0  # F_2 = GF(2)
_SELECT_BLOCK = 128

# `frx.default_backend()` reports the CUDA backend as `"gpu"`, not `"cuda"`.
#
# The accelerated select-XOR paths branch on which backend, not merely on "not
# CPU", and this comment is the one place that says why — the branches
# themselves only note what is local to them.
#
# The two backends differ in *route*: CUDA reaches its kernels through Pallas
# plus one hand kernel behind FFI, Metal through plugin custom calls only,
# having no Pallas at all. They also differ in *coverage*, which is what makes
# the branches asymmetric rather than a single "is accelerated" test. Metal has
# kernels for the unbatched 128-bit `elements` reduce and for the packed-byte
# gather; it takes the portable oracle for the batched stack, the non-128
# widths, and the unpacked 0/1 selector matrix. Anything that is neither
# backend takes the portable oracle throughout.
_CUDA = "gpu"
_METAL = "metal"

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


_ELEMENTS_TARGET_PROGRAMS = 8192


def _elements_grid(n: int, block: int) -> tuple[int, int, int, int]:
    """The `n`-parallel grid plan both elements kernels share:
    `(iters, rows_per_program, programs, padded_n)`.

    Gridding on the tiny `(W, L)` output would give only `W*L` programs, each
    serial over `n` — at flock m=28 that is 512 programs looping 16384 blocks,
    ~18x slower than the sibling `bits` kernel purely from under-parallelization.
    Instead grid over `n`: each program folds `iters` `block`-row steps into one
    accumulator, reading every selector/value once; the per-program partials are
    then XOR-combined. `_ELEMENTS_TARGET_PROGRAMS` bounds the partials array so
    the large-`m` shapes (#504) stay in memory."""
    iters = max(1, -(-n // (block * _ELEMENTS_TARGET_PROGRAMS)))
    rows_per_program = block * iters
    programs = -(-n // rows_per_program)
    padded_n = programs * rows_per_program
    return iters, rows_per_program, programs, padded_n


def _bit_select_reduce_elements_pallas(
    selectors: Array, values: Array, width: int, limbs: int
) -> Array:
    """Pallas lowering for `(n,) x (n,) -> (W, L)` limb output, parallel over n.

    Each step reads a `block`-row tile of both streams into a `(block, W, L)`
    accumulator, folded to `(W, L)` once per program by [`_xor_reduce_rows`]:
    the in-loop update must stay elementwise, since per-row indexing does not
    lower on Pallas GPU (`slice`) and an in-loop parity fold is shuffle-bound.
    """
    n = selectors.shape[0]
    # block=2 balances load width against accumulator register pressure
    # (measured optimum on the RTX 5090 target); wider tiles stream faster only
    # with the accumulate removed, so the constraint is acc-stream interaction,
    # not bandwidth.
    block = 2
    iters, rows_per_program, programs, padded_n = _elements_grid(n, block)
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))
    values = fnp.pad(values, ((0, padded_n - n), (0, 0)))

    def kernel(selector_ref: Any, value_ref: Any, out_ref: Any) -> None:
        program = pl.program_id(0)
        shifts = fnp.arange(_LIMB_BITS, dtype=_LIMB)
        base = program * rows_per_program + fnp.arange(block, dtype=fnp.int32)

        def body(step: Array, acc: Array) -> Array:
            rows = base + step * block
            sel = selector_ref[rows, :]
            val = value_ref[rows, :]
            bits = ((sel[:, :, None] >> shifts) & 1).reshape(block, width)
            # 0/1 bits -> 0/~0 masks: the AND select keeps the hot loop in the
            # LOP3 class instead of integer multiply, and a padded zero row
            # selects nothing.
            mask = _LIMB(0) - bits
            return acc ^ (mask[:, :, None] & val[:, None, :])

        acc = lax.fori_loop(0, iters, body, fnp.zeros((block, width, limbs), _LIMB))
        out_ref[program] = _xor_reduce_rows(acc)

    partials = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((programs, width, limbs), _LIMB),
        grid=(programs,),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_elements",
    )(selectors, values)
    return lax.reduce_xor(partials, (0,))


def _bit_select_reduce_elements_ffi(selectors_l: Array, values_l: Array) -> Array:
    """Plugin custom-call lowering for the `(n,) x (n,) -> (W, L)` reduce at
    `W = 128`.

    CUDA and Metal both register a handler for this target and XLA selects by
    platform, so this is one lowering rather than two. The call site still has
    to name those platforms: an unregistered one raises rather than falling
    back to the portable expression. Each hand kernel
    holds the XOR accumulator in registers while both operand streams pass
    through on-chip scratch — shared memory on CUDA, threadgroup memory on
    Metal — decoupling the accumulate from the load stream, which the Pallas
    lowerings cannot express (Triton has no scratch memory; Mosaic-GPU cannot
    partition rows across warps). XOR commutes, so the result is byte-identical
    to [`_bit_select_reduce_elements_pallas`] for any kernel grid.
    """
    return frx.ffi.ffi_call(
        "frx_bit_select_xor_reduce_elements",
        frx.ShapeDtypeStruct((128, 4), _LIMB),
    )(selectors_l, values_l)


def _bit_select_packed_bytes_ffi(selectors: Array, values: Array, limbs: int) -> Array:
    """Plugin custom-call lowering for packed-byte `(n, B/8) x (B,) -> (n, L)`.

    Drop-in for [`_bit_select_packed_bytes_pallas`], down to sharing its
    [`_byte_xor_table`]: `reduce="bits"` and [`byte_select_xor_reduce`] are the
    same reduction over the bit axis, so one kernel serves both.
    """
    table = _byte_xor_table(values, selectors.shape[1], limbs)
    return frx.ffi.ffi_call(
        "frx_bit_select_xor_reduce_packed_bytes",
        frx.ShapeDtypeStruct((selectors.shape[0], limbs), _LIMB),
    )(selectors, table)


def _bit_select_reduce_elements_batched_pallas(
    selectors: Array, values: Array, width: int, limbs: int
) -> Array:
    """Batched `(n,) selectors x (n, N) values -> (N, W, L)` limb output.

    The single-claim [`_bit_select_reduce_elements_pallas`] with the selectors —
    the shared packed witness — and their bit-decomposition read once per row and
    XOR-accumulated against all `N` value vectors, so a batched ring-switch open
    reads the witness once instead of once per claim. Parallel over `n`; the
    per-program partials are XOR-combined.

    `values` is selectors-major `(n, N, L)`: the shared row axis leads so the
    device load is a leading-axis gather (`value_ref[rows]`), the one indexing
    form the Triton lowering handles here. The batch `N` rides as a kernel-array
    axis, which the lowering needs power-of-2 sized, so `N` is padded up to the
    next power of two with zero value rows (a zero row XOR-reduces to zero) and
    sliced back. `N` a power of two (the flock two-claim open) pads to itself."""
    n, batch, _ = values.shape
    batch_pad = 1 << (batch - 1).bit_length()
    # One row per step: the batch axis already supplies the register-level
    # width the single-claim kernel gets from its row tile; widening both
    # regresses.
    block = 1
    iters, rows_per_program, programs, padded_n = _elements_grid(n, block)
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))
    # Pad rows (axis 0) to the grid and the batch (axis 1) up to a power of two.
    values = fnp.pad(values, ((0, padded_n - n), (0, batch_pad - batch), (0, 0)))

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
            bits = ((sel[:, :, None] >> shifts) & 1).reshape(width)
            val = value_ref[rows, :, :].reshape(batch_pad, limbs)
            # (W,) bit-select each of the N value rows -> (N, W, L).
            return acc ^ bits[None, :, None] * val[:, None, :]

        out_ref[program] = lax.fori_loop(
            0, iters, body, fnp.zeros((batch_pad, width, limbs), _LIMB)
        )

    partials = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((programs, batch_pad, width, limbs), _LIMB),
        grid=(programs,),
        compiler_params=plgpu.CompilerParams(),
        name="bit_select_xor_reduce_elements_batched",
    )(selectors, values)
    return lax.reduce_xor(partials, (0,))[:batch]


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
    # Keep all storage limbs in the same program.  Splitting limbs across the
    # grid rereads every selector byte once per limb, paying one global-memory
    # pass per limb.  A 64-row tile exposes enough blocks to saturate the GPU
    # while reusing each selector load across all limbs.
    block_rows = 64
    padded_n = ((n + block_rows - 1) // block_rows) * block_rows
    selectors = fnp.pad(selectors, ((0, padded_n - n), (0, 0)))

    table = _byte_xor_table(values, n_bytes, limbs)

    def kernel(selector_ref: Any, table_ref: Any, out_ref: Any) -> None:
        block = pl.program_id(0)
        rows = block * block_rows + fnp.arange(block_rows, dtype=fnp.int32)
        acc = fnp.zeros((block_rows, limbs), dtype=_LIMB)
        for byte_index in range(n_bytes):
            value = selector_ref[rows, byte_index].astype(fnp.int32)
            acc ^= table_ref[byte_index, value, :]
        out_ref[rows, :] = acc

    out = pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((padded_n, limbs), _LIMB),
        grid=(padded_n // block_rows,),
        compiler_params=plgpu.CompilerParams(num_warps=2),
        name="bit_select_xor_reduce_packed_bytes",
    )(selectors, table)
    return out[:n]


def _byte_xor_table(values: Array, n_bytes: int, limbs: int) -> Array:
    """Build the `(n_bytes, 256, limbs)` per-byte-position XOR table."""
    byte_values = values.reshape(n_bytes, 8, limbs)
    byte = fnp.arange(256, dtype=_LIMB)
    shift = fnp.arange(8, dtype=_LIMB)
    selected = ((byte[:, None] >> shift[None, :]) & 1).astype(_LIMB)
    return lax.reduce_xor(selected[None, :, :, None] * byte_values[:, None, :, :], (2,))


def _bit_select_wide_packed_bytes_pallas(
    selectors: Array, values: Array, limbs: int
) -> Array:
    """Packed-byte selection with one lane-parallel program per output row."""
    n, n_bytes = selectors.shape
    # Eight adjacent gathers per warp give the L2 much better locality than a
    # 128-wide vector on this table-shaped access.  Keep the 256 steps as a
    # loop: unrolling them saves only ~10% kernel time but makes full-prover
    # compilation prohibitively large.
    lanes = 8
    if n_bytes % lanes:
        pad = lanes - n_bytes % lanes
        selectors = fnp.pad(selectors, ((0, 0), (0, pad)))
        values = fnp.pad(values, ((0, pad * 8), (0, 0)))
    padded_bytes = selectors.shape[1]

    table = _byte_xor_table(values, padded_bytes, limbs)

    def kernel(selector_ref: Any, table_ref: Any, out_ref: Any) -> None:
        row = pl.program_id(0)
        lane = fnp.arange(lanes, dtype=fnp.int32)

        def body(step: Array, acc: Array) -> Array:
            byte_index = step * lanes + lane
            value = selector_ref[row, byte_index].astype(fnp.int32)
            return acc ^ table_ref[byte_index, value, :]

        acc = lax.fori_loop(
            0,
            padded_bytes // lanes,
            body,
            fnp.zeros((lanes, limbs), dtype=_LIMB),
        )
        out_ref[row, :] = _xor_reduce_rows(acc)

    return pl.pallas_call(
        kernel,
        out_shape=frx.ShapeDtypeStruct((n, limbs), _LIMB),
        grid=(n,),
        compiler_params=plgpu.CompilerParams(num_warps=1),
        name="bit_select_xor_reduce_wide_packed_bytes",
    )(selectors, table)


@frx.jit
def byte_select_xor_reduce(selectors: Array, values: Array) -> Array:
    """XOR weights selected by row-major packed bytes.

    `selectors` is a `uint8` matrix `(n, B/8)`, where byte `j` stores selector
    bits `8*j .. 8*j+7` least-significant-bit first. `values` is a binary-field
    vector `(B,)`; the result `(n,)` XORs `values[b]` wherever row bit `b` is set.

    Where a kernel exists, a 256-entry XOR table per byte position feeds the
    gather, matching flock-core's `UniSkipFoldTable`: the kernel reads `B/8`
    selector bytes per row and never expands the `(n, B)` bit matrix. Backends
    without one keep a compact source-level expression as the portable oracle.
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
    backend = frx.default_backend()
    if backend == _CUDA:
        if selectors.shape[1] <= 16:
            out_l = _bit_select_packed_bytes_pallas(selectors, values_l, limbs)
        else:
            out_l = _bit_select_wide_packed_bytes_pallas(selectors, values_l, limbs)
    elif backend == _METAL:
        # One kernel covers every row width, so Metal needs no wide/narrow
        # split: the gather is `S` table loads per row either way.
        out_l = _bit_select_packed_bytes_ffi(selectors, values_l, limbs)
    else:
        shifts = fnp.arange(8, dtype=fnp.uint8)
        unpacked = ((selectors[:, :, None] >> shifts) & 1).reshape(
            selectors.shape[0], bits
        )
        out_l = lax.reduce_xor(
            unpacked[:, :, None].astype(_LIMB) * values_l[None, :, :], (1,)
        )
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

    `reduce="elements"` also batches: a selectors-major `values` of shape
    `(n, N)` reduces all `N` columns against the shared selectors in one pass —
    the result is `(N, W)`, row `k` the reduction against `values[:, k]`. A
    batched ring-switch open uses this so the packed witness (the selectors) is
    read once for all `N` claims instead of once per claim.

    For `reduce="bits"`, selectors may instead be an explicit Boolean/integer
    0/1 matrix of shape `(n, B)` with `values.shape == (B,)`. This is the form
    used when a consumer already holds unpacked witness rows.

    Where a kernel exists, the lowering streams directly into the limb output.
    It never materializes the broadcast `(n, W, L)` selection; its working set
    is one fixed-size register block. Backends without one use the compact
    source-level expression so the primitive remains portable and easy to
    validate.
    """
    if selectors.ndim not in (1, 2):
        raise ValueError(
            "bit-select XOR reduction expects 1D packed or 2D unpacked selectors"
        )
    if values.ndim not in (1, 2):
        raise ValueError(
            "bit-select XOR reduction expects 1D values, or a 2D (n, N) stack "
            'for batched reduce="elements"'
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
        # Metal is deliberately absent: its accelerated lowering gathers packed
        # selector bytes, and these selectors are an explicit 0/1 matrix whose
        # bit count need not be a multiple of 8. Packing it to reach the kernel
        # would cost a pass over the (n, B) matrix the kernel exists to avoid
        # materializing, so Metal keeps the portable expression here.
        if frx.default_backend() == _CUDA:
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
        # `values` is `(n,)` for one claim or a selectors-major `(n, N)` stack
        # for N claims that share the selectors (the ring-switch batched-open
        # path: the packed witness is read once for all N). The reduced axis —
        # the one matching the selectors — leads in both.
        batched = values.ndim == 2
        if values.shape[0] != selectors.shape[0]:
            raise ValueError(
                f"values must match selectors on the reduced (leading) axis for "
                f'reduce="elements": {values.shape} vs {selectors.shape}'
            )
        backend = frx.default_backend()
        # One lowering, not two: both backends register a handler for this
        # target and XLA selects by platform. The set stays explicit because an
        # unregistered platform raises rather than falling back, so it has to
        # name exactly who has a handler. Metal appears in no arm below, which
        # is why its batched stacks and non-128 widths take the portable
        # expression until one is measured to matter.
        if backend in (_CUDA, _METAL) and not batched and width == 128:
            out_l = _bit_select_reduce_elements_ffi(selectors_l, values_l)
        elif backend == _CUDA:
            elements_pallas = (
                _bit_select_reduce_elements_batched_pallas
                if batched
                else _bit_select_reduce_elements_pallas
            )
            out_l = elements_pallas(selectors_l, values_l, width, limbs)
        else:
            bits = _bits(selectors)  # (n, width)
            if batched:
                # (n, width) x (n, N, limbs) -> (N, width, limbs)
                out_l = lax.reduce_xor(
                    bits[:, None, :, None] * values_l[:, :, None, :], (0,)
                )
            else:
                out_l = lax.reduce_xor(bits[:, :, None] * values_l[:, None, :], (0,))
        return _from_limbs(out_l, values.dtype)

    if reduce == "bits":
        if values.shape != (width,):
            raise ValueError(
                f'values must have shape ({width},) for reduce="bits", got '
                f"{values.shape}"
            )
        backend = frx.default_backend()
        if backend in (_CUDA, _METAL):
            selector_bytes = lax.bitcast_convert_type(selectors_l, fnp.uint8).reshape(
                selectors.shape[0], width // 8
            )
            packed_bytes = (
                _bit_select_packed_bytes_pallas
                if backend == _CUDA
                else _bit_select_packed_bytes_ffi
            )
            out_l = packed_bytes(selector_bytes, values_l, limbs)
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
