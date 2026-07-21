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

import frx
import frx.numpy as fnp
from frx import Array, jit

from zorch.utils import binary_field as bf


@jit
def bit_slice_evals(packed_witness: Array, tensor: Array) -> Array:
    """`s_hat_v[r] = Σ_i bit_r(packed_witness[i]) · tensor[i]` for `r ∈ [0, W)`.

    `(n,) × (n,) -> (W,)`. The memory-bounded bit-select reduction accumulates
    directly into the output without materializing its `(n, W, L)` broadcast.
    """
    return bf.bit_select_xor_reduce(packed_witness, tensor, reduce="elements")


@jit
def rs_eq_ind(tensor: Array, eq_r_dprime: Array) -> Array:
    """`rs_eq_ind[i] = Σ_b bit_b(tensor[i]) · eq_r_dprime[b]` — the transparent
    weight vector of the reduced claim (the same select-XOR kernel as
    [`bit_slice_evals`] with the bit-decomposed side swapped).

    `(n,) × (W,) -> (n,)`.
    """
    return bf.bit_select_xor_reduce(tensor, eq_r_dprime, reduce="bits")


@jit
def tensor_algebra_transpose(v: Array) -> Array:
    """The `W×W` bit transpose between the two readings of a tensor-algebra
    element: `bit_b(out[h]) = bit_h(v[b])`. `(W,) -> (W,)`; an involution."""
    w = bf.field_bit_width(v.dtype)
    if v.shape != (w,):
        # Any other length reshapes into a rectangular bit matrix and returns
        # garbage instead of erroring.
        raise ValueError(
            f"a tensor-algebra element over {fnp.dtype(v.dtype).name} is "
            f"shape ({w},), got {v.shape}"
        )
    bits_t = bf._bits(v).T  # (W, W): row h = bit h of every v[b]
    return bf._from_limbs(bf._limbs_from_bits(bits_t), v.dtype)


@jit
def inner_product(a: Array, b: Array) -> Array:
    """`Σ_i a[i]·b[i]` — field multiplies then a native field-additive reduce.
    `(n,) × (n,) -> ()`."""
    return fnp.sum(a * b, axis=0)


@partial(
    frx.tree_util.register_dataclass,
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
    s_hat_v: Array, suffix_tensor: Array, eq_r_dprime: Array
) -> RingSwitch:
    """The prover-side reduction for one claim.

    `s_hat_v` is the wire message from [`bit_slice_evals`] — taken as input, not
    recomputed from the witness, so a caller cannot assemble the reduction
    without first holding the message it must observe (and the dominant kernel
    runs once). `suffix_tensor` is the eq tensor of the claim point's suffix
    (the coordinates that index packed elements), `eq_r_dprime` the eq tensor of
    the post-observe challenge `r''` (length `W`). Sampling order stays the
    caller's — see the module docstring.
    """
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
    # 1 ⊗ 1 by concatenation — `.at[].set` and `pad` don't legalize on these
    # dtypes.
    e = fnp.concatenate([fnp.ones((1,), dtype), fnp.zeros((w - 1,), dtype)])
    for i in range(z_vals.shape[0]):
        vert = e * z_vals[i]
        horiz = tensor_algebra_transpose(tensor_algebra_transpose(e) * query[i])
        e = e + vert + horiz
    return inner_product(tensor_algebra_transpose(e), eq_r_dprime)
