# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reed-Solomon over a binary field's additive-NTT domain; implements
FoldableCode.

A binary field has no power-of-two multiplicative subgroup (the unit group's
order is odd) and `x == -x` in characteristic 2, so `ReedSolomon`'s
`(x, -x)`-conjugate FRI geometry does not exist there. The additive-NTT
codeword (`lax.ntt` lowers to the LCH transform for binary-field dtypes,
evaluating the message in the novel polynomial basis of Lin–Chung–Han 2014,
https://arxiv.org/abs/1404.3458) folds on the F2-linear subspace structure
instead: layer entries pair ADJACENTLY — `(2p, 2p+1)` folds to `p` — and the
fold is the inverse-butterfly line evaluation

    v = u_in + v_in ;  u = u_in + v * t ;  out = u + beta * (u + v)

with `t` the layer's subspace-polynomial twiddle. Each fold halves the degree
bound, and the fully folded honest codeword is a constant (`check_final`).
The twiddle schedule is anchored to THIS code's `block_len` — layer `l` uses
the subspace polynomial `log_d - l - 1` levels from the top — so intermediate
layers live on projected subspace domains, NOT on a smaller standard-basis
code's domain: a fresh `AdditiveReedSolomon(message_len // 2, blowup)` is a
different code from the folded layer, and every layer of one fold chain must
go through the one instance that committed the fresh codeword.

The twiddle table is the normalized subspace-polynomial evaluation table of
the standard basis {1, x, ..., x^(log_d-1)}: rows follow the recurrence
`W_i(z) = W_{i-1}(z) * (W_{i-1}(z) + W_{i-1}(b_{i-1}))`, each normalized by
`W_i(b_i)`; layer `l`'s twiddles are the subset-XOR span of row
`log_d - l - 1`, laid out layer-major so a layer with `half` pairs reads
`[half - 1, 2*half - 1)`. Built on device at construction (binary-field
division is a compiler lowering) by the same log-doubling idiom as
`ReedSolomon._coset_powers`, and memoized per `(log_d, dtype)` — the table is
data-independent and the row inverses are not free.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as jnp
import numpy as np
from frx import Array, lax

from zorch.utils.bits import log2_strict_usize


def _monomial_basis(log_d: int, dtype: Any) -> Array:
    """The storage bit-patterns of the basis {1, x, ..., x^(log_d-1)} — basis
    element `i` is storage bit `i` for any binary-field dtype."""
    itemsize = np.dtype(dtype).itemsize
    if log_d > itemsize * 8:
        raise ValueError(f"log_d {log_d} exceeds {np.dtype(dtype)} storage")
    buf = np.zeros((log_d, itemsize), np.uint8)
    for i in range(log_d):
        buf[i, i // 8] = 1 << (i % 8)
    return jnp.asarray(np.frombuffer(buf.tobytes(), dtype=dtype))


_TWIDDLE_CACHE: dict[tuple[int, str], Array] = {}


def additive_ntt_twiddles(log_d: int, dtype: Any) -> Array:
    """The layer-major additive-NTT twiddle table, `[2^log_d - 1]` over
    `dtype`. Layer `l` (with `2^l` blocks) occupies `[2^l - 1, 2^(l+1) - 1)`."""
    key = (log_d, str(np.dtype(dtype)))
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached

    # Characteristic-2 gate, probed semantically (1 + 1 == 0) rather than by
    # dtype name — the additive-NTT subspace geometry only exists there.
    one = np.ones(1, dtype=dtype)
    if not (one + one == np.zeros(1, dtype=dtype)).all():
        raise TypeError(
            f"{np.dtype(dtype).name} is not a binary field; the additive-NTT"
            " domain needs characteristic 2"
        )

    if log_d == 0:
        table = jnp.zeros((0,), dtype)
        _TWIDDLE_CACHE[key] = table
        return table

    basis = _monomial_basis(log_d, dtype)
    rows = [basis]
    for _ in range(1, log_d):
        prev = rows[-1]
        rows.append(prev[1:] * (prev[1:] + prev[0]))
    rows = [row / row[0] for row in rows]

    parts = [jnp.zeros((1,), dtype)]
    for layer in range(1, log_d):
        span_basis = rows[log_d - layer - 1][1:]  # length == layer
        cur = jnp.zeros((1,), dtype)
        for j in range(layer):
            cur = jnp.concatenate([cur, cur + span_basis[j]])
        parts.append(cur)
    table = jnp.concatenate(parts)
    _TWIDDLE_CACHE[key] = table
    return table


class AdditiveReedSolomon:
    """Reed-Solomon code over a binary field's additive-NTT (LCH) domain;
    implements FoldableCode. `block_len = message_len * blowup`."""

    def __init__(self, message_len: int, blowup: int, dtype: Any) -> None:
        log2_strict_usize(message_len)
        log2_strict_usize(blowup)
        self.message_len = message_len
        self.block_len = message_len * blowup
        self.dtype = dtype
        log_d = log2_strict_usize(self.block_len)
        self._twiddles = additive_ntt_twiddles(log_d, dtype)

    # Value equality/hash for static jit-zone keys — the LinearCode seam
    # contract (#214).
    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.message_len,
            self.block_len,
            str(np.dtype(self.dtype)),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AdditiveReedSolomon) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def encode(self, message: Array) -> Array:
        if message.shape[-1] != self.message_len:
            raise ValueError(
                f"message last axis must be {self.message_len}, "
                f"got {message.shape[-1]}"
            )
        tail = message.shape[:-1] + (self.block_len - self.message_len,)
        coeffs = jnp.concatenate([message, jnp.zeros(tail, self.dtype)], axis=-1)
        return lax.ntt(coeffs, ntt_type="NTT", ntt_length=self.block_len)

    def fold(self, codeword: Array, beta: Array) -> Array:
        half = codeword.shape[0] // 2
        t = self._twiddles[half - 1 : 2 * half - 1]
        pairs = codeword.reshape(half, 2)
        u_in, v_in = pairs[:, 0], pairs[:, 1]
        v = u_in + v_in
        u = u_in + v * t
        return u + beta * (u + v)

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        half = self.block_len >> (level + 1)
        t = self._twiddles[half - 1 + positions]
        v = lo + hi
        u = lo + v * t
        return u + beta * (u + v)

    def pair_leaves(self, codeword: Array) -> Array:
        return codeword.reshape(-1, 2)

    def pair_indices(self, positions: Array, level: int) -> tuple[Array, Array]:
        return 2 * positions, 2 * positions + 1

    def layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        indices = []
        q = positions
        for _ in range(num_rounds):
            q = q // 2
            indices.append(q)
        return indices

    def check_final(self, final: Array, claim: Array) -> Array:
        return jnp.all(final == claim)
