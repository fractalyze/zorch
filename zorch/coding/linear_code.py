# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LinearCode seam every encoding builds on.

A linear code maps a length-`message_len` message to a length-`block_len`
codeword over a single field dtype (`block_len > message_len`; the rate is
`message_len / block_len`). `encode` acts on the last axis, so leading batch
axes — many polynomials, or a matrix of rows — ride through untouched.
Reed-Solomon is one implementation; any other linear code (Brakedown, ...)
drops in unchanged.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface, like the Permutation seam: a code seats in static jit-zone
keys (inside provers/verifiers passed as static args), where identity equality
silently re-traces the zone on every freshly built same-config instance
(#214). A Protocol cannot enforce this — each implementation carries it
(`ReedSolomon`, `BitReversedReedSolomon`).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from frx import Array


@runtime_checkable
class LinearCode(Protocol):
    message_len: int  # k, the message dimension
    block_len: int  # n, the codeword length (n > k; rate = k / n)
    dtype: Any  # field dtype of both message and codeword

    def encode(self, message: Array) -> Array:
        """Encode `(..., message_len)` to `(..., block_len)` over `dtype`."""
        ...
