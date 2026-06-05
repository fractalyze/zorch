# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reed-Solomon as a FoldableCode: low-degree extension via the native NTT.

`encode` reads the message as the `message_len` low-order coefficients of a
polynomial, zero-pads to `block_len`, and evaluates it on the order-`block_len`
two-adic subgroup (or a coset of it). The evaluation is `jax.lax.fft` — the
ZKX-native NTT — which lowers to one fused kernel and auto-decomposes extension
fields into prime-field NTTs.

There is deliberately no hand-rolled butterfly: a `jnp` butterfly would be
log(n) unfused kernels the compiler cannot recognize as an NTT. Reed-Solomon
hands its evaluation to the native op, the way poseidon2 hands its algebra to
zkx rather than fusing it by pattern-match.

`fri_fold` is the codeword fold shared by every FRI-style scheme (FRI,
Basefold, WHIR, STARK); the fold half of the seam delegates to it. It lives
in this module so the fold's x-coordinates stay the *same* evaluation domain
the encoder used. WHIR's k-ary generalization is deferred to its first
consumer.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import zk_dtypes
from jax import Array, lax

from zorch.utils.bits import is_power_of_two


def _base_dtype(dtype: Any) -> Any:
    try:
        return zk_dtypes.efinfo(dtype).base_field_dtype
    except ValueError:
        return dtype


def eval_domain(dtype: Any, n: int) -> Array:
    """The order-`n` two-adic subgroup points [d₀..d_{n-1}] in `lax.fft` order.

    `lax.fft` of the coefficient vector of p(X)=X (i.e. e₁) returns
    [p(d₀)..p(d_{n-1})] = [d₀..d_{n-1}], so the domain is read off the same NTT
    the encoder uses. `n` must be a power of two; the order-1 subgroup is {1}."""
    if not is_power_of_two(n):
        raise ValueError(f"eval_domain size must be a power of two, got {n}")
    if n == 1:
        return jnp.ones((1,), dtype)
    e1 = jnp.zeros(n, dtype).at[1].set(jnp.ones((), dtype))
    return lax.fft(e1, "FFT", n)


class ReedSolomon:
    """Reed-Solomon code over `dtype`; implements FoldableCode.

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
    ) -> None:
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

    def fold(self, codeword: Array, beta: Array) -> Array:
        """FoldableCode fold: natural-order `(x, -x)` conjugate pairs."""
        return fri_fold(codeword, beta)

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened pairs of layer `level`; the x-coordinates are the first
        half of the layer's (level-times-squared) evaluation domain."""
        domain = eval_domain(self.dtype, self.block_len >> level)
        return fri_fold_values(lo, hi, beta, domain[positions])

    def check_final(self, final: Array, claim: Array) -> Array:
        """A message-length-1 RS codeword is the constant polynomial, so base-code
        membership and message == `claim` collapse into one comparison."""
        return jnp.all(final == claim)


def fri_fold_values(fx: Array, fnx: Array, beta: Array, x: Array) -> Array:
    """g(x²) = (f(x)+f(−x))/2 + β·(f(x)−f(−x))/(2x). f-values may be EF, x in BF."""
    one = jnp.ones((), fx.dtype)
    two = one + one
    return (fx + fnx) / two + beta * (fx - fnx) / (two * x)


def fri_fold(codeword: Array, beta: Array) -> Array:
    """FRI-fold a natural-order RS codeword (length 2^m) by β, halving its length.

    Natural order: dⱼ and d_{j+n/2} = −dⱼ are conjugates, so f(x)=codeword[:half],
    f(−x)=codeword[half:], x=domain[:half]. Result is the fold over the order-(n/2)
    subgroup (=squared domain), again in natural order."""
    n = codeword.shape[0]
    if n < 2:
        raise ValueError(f"fri_fold requires a codeword of length >= 2, got {n}")
    half = n // 2
    domain = eval_domain(_base_dtype(codeword.dtype), n)
    return fri_fold_values(codeword[:half], codeword[half:], beta, domain[:half])
