# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ring-switching reduction ([DP24] §1.3) for binary-field PCS packing.

A prover over GF(2) stores its witness as raw bits but commits the *packed*
form: `W` bits per big-field element, so the PCS sees a multilinear with
`log2(W)` fewer variables. An IOP claim `ẑ(point) = v` about the bit-coefficient
multilinear `ẑ` therefore cannot be opened directly — packing lays bits out as
big-field *coordinates*, not evaluations. Ring switching bridges the gap with
one wire vector and one challenge:

  * `s_hat_v[r] = Σ_i bit_r(witness[i]) · tensor[i]` — the `W` bit-slice partial
    evaluations of `ẑ` at the claim point's suffix (`tensor` is its eq tensor).
    The only ring-switch data on the wire; the verifier checks the incoming
    claim against it with its own prefix weights (a consumer concern — see
    "what stays outside" below).
  * After observing `s_hat_v`, the verifier samples `r'' ∈ F^{log2(W)}`.
    Collapsing the bit axis with `eq(r'')` turns the claim into

        Σ_i  packed_witness[i] · rs_eq_ind[i]  =  ⟨transpose(s_hat_v), eq(r'')⟩

    whose left side is a transparent-weighted sum over the *packed* multilinear
    — exactly the shape a Basefold-style PCS opens — and whose right side the
    verifier computes from the wire alone.

Both readings are the two axis-orders of one tensor-algebra element
`Σ_i tensor[i] ⊗ witness[i] ∈ F ⊗_{GF(2)} F` (a `W×W` bit matrix); converting
between them is [`tensor_algebra_transpose`], a pure bit transpose. Soundness
rides on `r''` being sampled *after* `s_hat_v` is observed — see the contract
below.

## The GF(2)-basis is the dtype's storage-bit basis

`bit_r` and the packing must decompose over the *same* GF(2)-basis of the field.
This module uses the dtype's little-endian storage bits (lane 0's bit 0 is
`r = 0`), which is the packing convention of a bitcast — so a consumer that
packs its bit witness by reinterpreting the bit buffer as field elements agrees
with these kernels by construction, in any representation (tower or GHASH
alike; the reduction is basis-blind, only claim-*check* weights are not).

## Transcript-free: the caller owns Fiat-Shamir order

These are pure functions; nothing here observes or samples. The caller must
enforce, per claim: observe `s_hat_v` → *then* sample `r''` (and, when batching
claims, sample the combination scalars only after every claim's `s_hat_v` is
observed — the reduction is linear, so batching is the caller scaling each
`rs_eq_ind` by its scalar). This keeps the block agnostic to the host's
transcript (a byte-duplex, a device sponge, or a native `Transcript`).

## What stays outside

  * Prefix claim-check weights: the incoming claim's prefix may live in a
    univariate-skip domain whose Lagrange weights are scheme-specific.
  * Serialization of `s_hat_v` and the transcript labels.
  * The PCS that opens the reduced claim (`pcs/basefold`, `pcs/ligerito`, …).

The verifier never materializes `rs_eq_ind` (length `2^ℓ`): [`eval_rs_eq`]
evaluates its MLE at the PCS's final point in `O(ℓ·W)` field multiplies plus
`O(ℓ)` bit transposes, polylog in the witness ([DP24] §1.3, Figure 3).

[DP24]: https://eprint.iacr.org/2024/504
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, jit, lax

_LANE = jnp.uint32
_LANE_BITS = 32


def field_bit_width(dtype: Any) -> int:
    """`W`: the field's size in storage bits (= its GF(2)-dimension)."""
    width = jnp.dtype(dtype).itemsize * 8
    if width % _LANE_BITS != 0:
        raise ValueError(
            f"{jnp.dtype(dtype).name} is {width} bits; the bit kernels lane over "
            f"uint{_LANE_BITS} and need a multiple of {_LANE_BITS}"
        )
    return width


def _lanes(x: Array) -> Array:
    """(...,) field -> (..., L) little-endian uint32 storage lanes."""
    return lax.bitcast_convert_type(x, _LANE)


def _from_lanes(lanes: Array, dtype: Any) -> Array:
    """(..., L) uint32 lanes -> (...,) field. Inverse of [`_lanes`]."""
    return lax.bitcast_convert_type(lanes, dtype)


def _bits(x: Array) -> Array:
    """(...,) field -> (..., W) 0/1 uint32: bit `r` is lane `r//32`'s bit `r%32`."""
    lanes = _lanes(x)
    shifts = jnp.arange(_LANE_BITS, dtype=_LANE)
    bits = (lanes[..., :, None] >> shifts) & _LANE(1)
    return bits.reshape(*x.shape, -1)


def _xor_reduce(lanes: Array, axis: int) -> Array:
    """XOR-reduce uint32 lanes — field addition of any binary field, done on the
    representation so it needs no dtype arithmetic support."""
    return lax.reduce(lanes, _LANE(0), lax.bitwise_xor, (axis,))


def add(a: Array, b: Array) -> Array:
    """Field addition via lane XOR — exact in every binary-field representation."""
    return _from_lanes(_lanes(a) ^ _lanes(b), a.dtype)


@jit
def bit_slice_evals(packed_witness: Array, tensor: Array) -> Array:
    """`s_hat_v[r] = Σ_i bit_r(packed_witness[i]) · tensor[i]` for `r ∈ [0, W)`.

    `(n,) × (n,) -> (W,)`. A bit-select + XOR-reduce — no field multiplies; the
    `(n, W, L)` intermediate stays inside the jit fusion.
    """
    w_bits = _bits(packed_witness)  # (n, W)
    t_lanes = _lanes(tensor)  # (n, L)
    acc = _xor_reduce(w_bits[:, :, None] * t_lanes[:, None, :], axis=0)  # (W, L)
    return _from_lanes(acc, tensor.dtype)


@jit
def rs_eq_ind(tensor: Array, eq_r_dprime: Array) -> Array:
    """`rs_eq_ind[i] = Σ_b bit_b(tensor[i]) · eq_r_dprime[b]` — the transparent
    weight vector of the reduced claim (the same select-XOR kernel as
    [`bit_slice_evals`] with the bit-decomposed side swapped).

    `(n,) × (W,) -> (n,)`.
    """
    t_bits = _bits(tensor)  # (n, W)
    eq_lanes = _lanes(eq_r_dprime)  # (W, L)
    acc = _xor_reduce(t_bits[:, :, None] * eq_lanes[None, :, :], axis=1)  # (n, L)
    return _from_lanes(acc, eq_r_dprime.dtype)


@jit
def tensor_algebra_transpose(v: Array) -> Array:
    """The `W×W` bit transpose between the two readings of a tensor-algebra
    element: `bit_b(out[h]) = bit_h(v[b])`. `(W,) -> (W,)`; an involution."""
    w = field_bit_width(v.dtype)
    if v.shape != (w,):
        # Any other length reshapes into a rectangular bit matrix and returns
        # garbage instead of erroring.
        raise ValueError(
            f"a tensor-algebra element over {jnp.dtype(v.dtype).name} is "
            f"shape ({w},), got {v.shape}"
        )
    bits_t = _bits(v).T  # (W, W): row h = bit h of every v[b]
    shifts = (_LANE(1) << jnp.arange(_LANE_BITS, dtype=_LANE))[None, None, :]
    lanes = jnp.sum(bits_t.reshape(w, -1, _LANE_BITS) * shifts, axis=-1, dtype=_LANE)
    return _from_lanes(lanes, v.dtype)


@jit
def inner_product(a: Array, b: Array) -> Array:
    """`Σ_i a[i]·b[i]` — field multiplies + a lane-XOR reduce. `(n,) × (n,) -> ()`."""
    return _from_lanes(_xor_reduce(_lanes(a * b), axis=0), a.dtype)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["s_hat_v", "rs_eq_ind", "claim"],
    meta_fields=[],
)
@dataclass(frozen=True)
class RingSwitch:
    """One reduced claim: observe `s_hat_v`, then let the PCS prove
    `Σ_i packed_witness[i] · rs_eq_ind[i] = claim`. A registered pytree so a
    consumer's jitted open path can return it across the `jit` boundary."""

    s_hat_v: Array  # (W,) — the wire
    rs_eq_ind: Array  # (2^ℓ,) — the PCS sumcheck's transparent counterpart
    claim: Array  # () — the reduced sumcheck target


def reduce_bit_claim(
    packed_witness: Array, suffix_tensor: Array, eq_r_dprime: Array
) -> RingSwitch:
    """The prover-side reduction for one claim.

    `suffix_tensor` is the eq tensor of the claim point's suffix (the coordinates
    that index packed elements), `eq_r_dprime` the eq tensor of the post-observe
    challenge `r''` (length `W`). Fiat-Shamir order is the caller's — see the
    module docstring.
    """
    if packed_witness.ndim != 1 or packed_witness.shape != suffix_tensor.shape:
        raise ValueError(
            f"packed_witness and suffix_tensor must share one 1-D shape, got "
            f"{packed_witness.shape} vs {suffix_tensor.shape}"
        )
    s_hat_v = bit_slice_evals(packed_witness, suffix_tensor)
    claim = inner_product(tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return RingSwitch(
        s_hat_v=s_hat_v,
        rs_eq_ind=rs_eq_ind(suffix_tensor, eq_r_dprime),
        claim=claim,
    )


def eval_rs_eq(z_vals: Array, query: Array, eq_r_dprime: Array) -> Array:
    """`MLE(rs_eq_ind)(query)` in `O(ℓ·W)` multiplies — the succinct verifier's
    replacement for materializing `rs_eq_ind` ([DP24] §1.3, Figure 3).

    `z_vals` are the claim-point suffix coordinates that built `suffix_tensor`,
    `query` the PCS's final challenge point (same length and variable order).
    Walks one tensor-algebra element `E`: in characteristic 2,
    `eq(z, q) = 1 + z + q`, so each step is `E += z·E (vertical) + q·E
    (horizontal)`; the final vertical fold against `eq(r'')` is the same
    transpose + inner product the prover-side claim uses.
    """
    if z_vals.shape != query.shape:
        raise ValueError(f"point length mismatch: {z_vals.shape} vs {query.shape}")
    (w,) = eq_r_dprime.shape
    dtype = eq_r_dprime.dtype
    one_lanes = jnp.zeros((w, jnp.dtype(dtype).itemsize // 4), dtype=_LANE)
    e = _from_lanes(one_lanes.at[0, 0].set(1), dtype)  # 1 ⊗ 1
    for i in range(z_vals.shape[0]):
        vert = e * z_vals[i]
        horiz = tensor_algebra_transpose(tensor_algebra_transpose(e) * query[i])
        e = add(e, add(vert, horiz))
    return inner_product(tensor_algebra_transpose(e), eq_r_dprime)
