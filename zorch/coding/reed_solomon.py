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

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import zk_dtypes
from jax import Array, lax

from zorch.utils.bits import is_power_of_two, log2_strict_usize

if TYPE_CHECKING:
    from zorch.coding.foldable_code import FoldableCode


def _base_dtype(dtype: Any) -> Any:
    try:
        return zk_dtypes.efinfo(dtype).base_field_dtype
    except ValueError:
        return dtype


def eval_domain(dtype: Any, n: int, *, shift: Array | None = None) -> Array:
    """The order-`n` two-adic subgroup points [d₀..d_{n-1}] in `lax.fft` order,
    or the coset points [shift·d₀..shift·d_{n-1}] when `shift` is given.

    `lax.fft` of the coefficient vector of p(X)=X (i.e. e₁) returns
    [p(d₀)..p(d_{n-1})] = [d₀..d_{n-1}], so the domain is read off the same NTT
    the encoder uses. `n` must be a power of two; the order-1 subgroup is {1}."""
    if not is_power_of_two(n):
        raise ValueError(f"eval_domain size must be a power of two, got {n}")
    if n == 1:
        domain = jnp.ones((1,), dtype)
    else:
        e1 = jnp.zeros(n, dtype).at[1].set(jnp.ones((), dtype))
        domain = lax.fft(e1, "FFT", n)
    return domain if shift is None else shift * domain


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
        self.coset_shift = coset_shift
        # Coset eval scales coeffs by [1, h, h^2, ..., h^{n-1}]; precompute it
        # once since h and n are fixed. Built by log-doubling — powers[m:2m] =
        # powers[:m] * h^m — because `jnp.arange` raises on extension dtypes
        # (iota unsupported) and a sequential `jnp.cumprod` is one
        # O(n)-depth kernel, paid eagerly at construction.
        self._coset_powers = None
        if coset_shift is not None:
            powers = jnp.ones((1,), dtype)
            step = jnp.asarray(coset_shift, dtype)
            while powers.shape[0] < self.block_len:
                powers = jnp.concatenate([powers, powers * step])
                step = step * step
            self._coset_powers = powers
        self._key: tuple | None = None

    # Value equality/hash for static jit-zone keys — the LinearCode seam
    # contract (#214). The key is cached host-side because jit dispatch
    # compares static args per call, and `tobytes` on the live coset-shift
    # array would cost a device->host sync each time (the Poseidon2Params
    # pattern).
    def _value_key(self) -> tuple:
        if self._key is None:
            shift = (
                None
                if self.coset_shift is None
                else np.asarray(self.coset_shift).tobytes()
            )
            self._key = (self.message_len, self.block_len, self.dtype, shift)
        return self._key

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, ReedSolomon):
            return NotImplemented
        return self._value_key() == other._value_key()

    def __hash__(self) -> int:
        return hash(self._value_key())

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
        if coeffs.ndim > 1 and _base_dtype(self.dtype) != self.dtype:
            # zkx's EF→base NTT decomposition rejects leading batch dims
            # (layout-assignment failure, zkx#637; `jax.vmap` lowers to the
            # same batched fft and fails identically): per-row 1-D NTTs until
            # that lands.
            flat = coeffs.reshape(-1, n)
            rows = lax.map(lambda row: lax.fft(row, "FFT", n), flat)
            return rows.reshape(coeffs.shape)
        return lax.fft(coeffs, "FFT", n)

    def domain(self) -> Array:
        """The points `encode` evaluates on, coset shift included."""
        return eval_domain(self.dtype, self.block_len, shift=self.coset_shift)

    def fold(self, codeword: Array, beta: Array) -> Array:
        """FoldableCode fold: natural-order `(x, -x)` conjugate pairs. The layer
        level — and with it the coset shift — is read off the codeword length."""
        level = log2_strict_usize(self.block_len // codeword.shape[0])
        return fri_fold(codeword, beta, shift=self._level_shift(level))

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened pairs of layer `level`; the x-coordinates are the first
        half of the layer's (level-times-squared) evaluation domain."""
        domain = eval_domain(
            self.dtype, self.block_len >> level, shift=self._level_shift(level)
        )
        return fri_fold_values(lo, hi, beta, domain[positions])

    def pair_leaves(self, codeword: Array) -> Array:
        """Natural order: conjugates sit a half-layer apart, so leaf `p` is
        `(codeword[p], codeword[p + half])`."""
        half = codeword.shape[0] // 2
        return jnp.stack([codeword[:half], codeword[half:]], axis=1)

    def check_final(self, final: Array, claim: Array) -> Array:
        """A message-length-1 RS codeword is the constant polynomial on any
        domain, so base-code membership and message == `claim` collapse into one
        comparison."""
        return jnp.all(final == claim)

    def pair_indices(self, positions: Array, level: int) -> tuple[Array, Array]:
        """Natural order: the conjugates of layer `level` sit a half-layer
        apart, and the lo index is the landing index itself."""
        return positions, positions + (self.block_len >> (level + 1))

    def layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        """Natural order: `a_i = q_i mod (n / 2^{i+1})` with `q_0 = positions`,
        `q_{i+1} = a_i`, elementwise over the query axis."""
        indices = []
        q = positions
        for i in range(num_rounds):
            a = q % (self.block_len >> (i + 1))
            indices.append(a)
            q = a
        return indices

    def _level_shift(self, level: int) -> Array | None:
        """Layer `level`'s domain shift, `coset_shift^(2^level)` — each fold
        lands on the squared domain, squaring the shift with it."""
        if self.coset_shift is None:
            return None
        shift = self.coset_shift
        for _ in range(level):
            shift = shift * shift
        return shift


def _bit_reverse_indices(positions: Array, n: int) -> Array:
    """Bit-reverse each index in `positions` within width `log2(n)` — the
    index-space mirror of `lax.bit_reverse`, for gathers too sparse to justify
    permuting the whole array."""
    bits = log2_strict_usize(n)
    rev = positions * 0
    for b in range(bits):
        rev = rev | (((positions >> b) & 1) << (bits - 1 - b))
    return rev


class BitReversedReedSolomon:
    """Reed-Solomon with codewords in bit-reversed evaluation order.

    Some commitment layouts store the codeword bit-reversed so a fold's point
    pair sits adjacently (`(2p, 2p+1)`) instead of a half-layer apart — Merkle
    paths of a pair then share all but their last node, and the layout is
    fold-stable (folding a bit-reversed layer yields the squared domain's
    codeword, again bit-reversed). The fold math is `ReedSolomon`'s; only the
    layout-dependent surfaces differ — pair geometry (`pair_indices` /
    `layer_positions`), the fold's x-coordinate gather, and the
    `encode`/`domain` output order.
    """

    def __init__(
        self,
        message_len: int,
        blowup: int,
        dtype: Any,
        *,
        coset_shift: Array | None = None,
    ) -> None:
        self._natural = ReedSolomon(message_len, blowup, dtype, coset_shift=coset_shift)
        self.message_len = message_len
        self.block_len = self._natural.block_len
        self.dtype = dtype

    # Value equality/hash via the wrapped natural-order code (see ReedSolomon);
    # the isinstance gate keeps the two layouts distinct.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitReversedReedSolomon):
            return NotImplemented
        return self._natural == other._natural

    def __hash__(self) -> int:
        return hash((BitReversedReedSolomon, self._natural))

    def encode(self, message: Array) -> Array:
        cw = self._natural.encode(message)
        return lax.bit_reverse(cw, dimensions=(cw.ndim - 1,))

    def domain(self) -> Array:
        """The points `encode` evaluates on, in codeword (bit-reversed) order."""
        return lax.bit_reverse(self._natural.domain(), dimensions=(0,))

    def fold(self, codeword: Array, beta: Array) -> Array:
        n = codeword.shape[0]
        if n < 2:
            raise ValueError(f"fold requires a codeword of length >= 2, got {n}")
        level = log2_strict_usize(self.block_len // n)
        pairs = codeword.reshape(n // 2, 2)
        return fri_fold_values(
            pairs[:, 0], pairs[:, 1], beta, self._pair_points(n, level)
        )

    def fold_values(
        self, lo: Array, hi: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        # Gather the few queried x-coordinates straight from the natural-order
        # half at bit-reversed indices — a full-width `lax.bit_reverse` of the
        # domain would be discarded except at `positions`.
        n = self.block_len >> level
        x = self._layer_domain(n, level)[_bit_reverse_indices(positions, n // 2)]
        return fri_fold_values(lo, hi, beta, x)

    def check_final(self, final: Array, claim: Array) -> Array:
        """Constant-polynomial membership is order-invariant."""
        return self._natural.check_final(final, claim)

    def pair_indices(self, positions: Array, level: int) -> tuple[Array, Array]:
        """Bit-reversed order: the pair landing at `positions` is adjacent."""
        return positions * 2, positions * 2 + 1

    def pair_leaves(self, codeword: Array) -> Array:
        """Bit-reversed order: conjugates are adjacent, so leaf `p` is the pair
        `(codeword[2p], codeword[2p + 1])`."""
        return codeword.reshape(codeword.shape[0] // 2, 2)

    def layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        """Bit-reversed order: each fold halves the index, `a_i = q >> (i+1)`."""
        indices = []
        q = positions
        for _ in range(num_rounds):
            q = q >> 1
            indices.append(q)
        return indices

    def _layer_domain(self, n: int, level: int) -> Array:
        """Layer `level`'s natural-order evaluation domain (length `n`)."""
        return eval_domain(
            _base_dtype(self.dtype), n, shift=self._natural._level_shift(level)
        )

    def _pair_points(self, n: int, level: int) -> Array:
        """x-coordinates of all of layer `level`'s pairs in pair order: entry
        `p` is the evaluation point of the pair `(2p, 2p+1)`, i.e. the natural
        domain's first half gathered through the bit-reversal. For a sparse
        gather see `fold_values`, which reverses the indices instead."""
        x = self._layer_domain(n, level)[: n // 2]
        if n > 2:
            x = lax.bit_reverse(x, dimensions=(0,))
        return x


def fri_fold_values(fx: Array, fnx: Array, beta: Array, x: Array) -> Array:
    """g(x²) = (f(x)+f(−x))/2 + β·(f(x)−f(−x))/(2x). f-values may be EF; x carries
    the domain's dtype."""
    one = jnp.ones((), fx.dtype)
    two = one + one
    return (fx + fnx) / two + beta * (fx - fnx) / (two * x)


def fri_fold(codeword: Array, beta: Array, *, shift: Array | None = None) -> Array:
    """FRI-fold a natural-order RS codeword (length 2^m) by β, halving its length.

    Natural order: dⱼ and d_{j+n/2} = −dⱼ are conjugates, so f(x)=codeword[:half],
    f(−x)=codeword[half:], x=domain[:half]. Result is the fold over the order-(n/2)
    squared domain, again in natural order.

    `shift` is the coset shift of the codeword's own domain. The fold lands on
    the squared domain, so the next layer's shift is `shift²` — iterating
    callers must square it each round."""
    n = codeword.shape[0]
    if n < 2:
        raise ValueError(f"fri_fold requires a codeword of length >= 2, got {n}")
    half = n // 2
    domain = eval_domain(_base_dtype(codeword.dtype), n, shift=shift)
    return fri_fold_values(codeword[:half], codeword[half:], beta, domain[:half])


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[FoldableCode] = ReedSolomon
    _bitrev: type[FoldableCode] = BitReversedReedSolomon
