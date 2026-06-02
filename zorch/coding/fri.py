# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""FRI-family codeword fold over the Reed-Solomon evaluation domain.

Shared by every FRI-style scheme (FRI, Basefold, WHIR, STARK); lives next to
`reed_solomon` so the fold's x-coordinates stay the *same* `lax.fft` evaluation
domain the encoder used — recovered via `eval_domain` (no field-generator table).
WHIR's k-ary generalization is deferred to its first consumer.
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
    half = n // 2
    domain = eval_domain(_base_dtype(codeword.dtype), n)
    return fri_fold_values(codeword[:half], codeword[half:], beta, domain[:half])
