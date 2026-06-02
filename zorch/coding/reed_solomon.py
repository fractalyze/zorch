# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reed-Solomon as a LinearCode: low-degree extension via the native NTT.

`encode` reads the message as the `message_len` low-order coefficients of a
polynomial, zero-pads to `block_len`, and evaluates it on the order-`block_len`
two-adic subgroup (or a coset of it). The evaluation is `jax.lax.fft` — the
ZKX-native NTT — which lowers to one fused kernel and auto-decomposes extension
fields into prime-field NTTs.

There is deliberately no hand-rolled butterfly: a `jnp` butterfly would be
log(n) unfused kernels the compiler cannot recognize as an NTT. Reed-Solomon
hands its evaluation to the native op, the way poseidon2 hands its algebra to
zkx rather than fusing it by pattern-match.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from zorch.utils.bits import is_power_of_two


class ReedSolomon:
    """Reed-Solomon code over `dtype`; implements LinearCode.

    `block_len = message_len * blowup` (both powers of two). With `coset_shift`
    set to a field element outside the subgroup, the codeword is the message
    polynomial evaluated on the coset `coset_shift * <subgroup>` rather than the
    subgroup itself — FRI/STARK want an evaluation domain disjoint from the
    trace domain. The shift is supplied by the caller, so the code carries no
    field-generator table.
    """

    def __init__(
        self,
        message_len: int,
        blowup: int,
        dtype: Any,
        *,
        coset_shift: Array | None = None,
    ):
        if not is_power_of_two(message_len):
            raise ValueError(f"message_len must be a power of two, got {message_len}")
        if not is_power_of_two(blowup):
            raise ValueError(f"blowup must be a power of two, got {blowup}")
        self.message_len = message_len
        self.block_len = message_len * blowup
        self.dtype = dtype
        # Coset eval scales coeffs by [1, h, h^2, ..., h^{n-1}]; precompute it
        # once since h and n are fixed. Built as a cumulative product because
        # `jnp.arange` raises on extension dtypes (iota unsupported), so the
        # exponent ramp cannot be formed the usual way.
        self._coset_powers = None
        if coset_shift is not None:
            seq = (
                jnp.full((self.block_len,), coset_shift, dtype)
                .at[0]
                .set(jnp.ones((), dtype))
            )
            self._coset_powers = jnp.cumprod(seq)

    def encode(self, message: Array) -> Array:
        if message.shape[-1] != self.message_len:
            raise ValueError(
                f"message last axis must be {self.message_len}, "
                f"got {message.shape[-1]}"
            )
        n = self.block_len
        tail = message.shape[:-1] + (n - self.message_len,)
        coeffs = jnp.concatenate([message, jnp.zeros(tail, self.dtype)], axis=-1)
        if self._coset_powers is not None:
            coeffs = coeffs * self._coset_powers
        return lax.fft(coeffs, "FFT", n)
