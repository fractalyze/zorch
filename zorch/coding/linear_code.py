# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LinearCode seam every encoding builds on.

A linear code maps a length-`message_len` message to a length-`block_len`
codeword over a single field dtype (`block_len > message_len`; the rate is
`message_len / block_len`). `encode` acts on the last axis, so leading batch
axes — many polynomials, or a matrix of rows — ride through untouched.
Reed-Solomon is one implementation; any other linear code (Brakedown, ...)
drops in unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class LinearCode(Protocol):
    message_len: int  # k, the message dimension
    block_len: int  # n, the codeword length (n > k; rate = k / n)
    dtype: Any  # field dtype of both message and codeword

    def encode(self, message: Array) -> Array:
        """Encode `(..., message_len)` to `(..., block_len)` over `dtype`."""
        ...
