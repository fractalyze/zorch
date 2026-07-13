# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""R1CS instance `(A·z) ∘ (B·z) = C·z` and the products the Spartan PIOP reduces.

The assignment is **witness-first** `z = (W, 1, X)`: `W` fills the low half, the
high half holds the constant `1` then the public inputs `X`. That layout is what
makes `r_y[0]` (the inner sumcheck's first-bound variable) the half-selector
between `W` and `(1, X)`, so `W` opens at `r_y[1:]`. Matrices are dense (a sparse
SPARK form would plug the same `matvecs` / combined-row interface). Field is the
caller's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class R1CS:
    """A dense R1CS instance `(A·z) ∘ (B·z) = C·z`.

    `a`, `b`, `c` are `(m, n)` dense matrices over the field; `m = num_cons` is a
    power of two and `n = 2·num_vars_padded` is a power of two with the witness in
    the low half (`z = (W, 1, X)`). `num_io` is the public-input count. The class
    holds no witness — an assignment `z` is passed to the product helpers, so one
    instance serves many assignments.
    """

    a: Array
    b: Array
    c: Array
    num_io: int

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape or self.a.shape != self.c.shape:
            raise ValueError("A, B, C must share shape")
        m, n = self.a.shape
        # power-of-two guards (log2_strict_usize raises on a non-power-of-two).
        log2_strict_usize(m)
        log2_strict_usize(n)
        if n % 2 != 0:
            raise ValueError("column count must be even (witness fills the low half)")
        if self.num_io >= n // 2:
            raise ValueError("num_io must fit in the high half after the constant 1")

    @property
    def num_cons(self) -> int:
        return self.a.shape[0]

    @property
    def num_cols(self) -> int:
        return self.a.shape[1]

    @property
    def num_vars_padded(self) -> int:
        """`|W|` slot count — the low half of `z`."""
        return self.num_cols // 2

    @property
    def s_x(self) -> int:
        """Outer-sumcheck variable count `log2(num_cons)`."""
        return log2_strict_usize(self.num_cons)

    @property
    def s_y(self) -> int:
        """Inner-sumcheck variable count `log2(num_cols)` (= `log2(|W|)+1`)."""
        return log2_strict_usize(self.num_cols)

    def matvecs(self, z: Array) -> tuple[Array, Array, Array]:
        """`(A·z, B·z, C·z)`, each length `num_cons` — the outer-sumcheck MLEs."""
        return self.a @ z, self.b @ z, self.c @ z

    def is_satisfied(self, z: Array) -> Array:
        """Row-wise `(A·z)∘(B·z) == C·z` for all rows (scalar bool)."""
        az, bz, cz = self.matvecs(z)
        return jnp.all(az * bz == cz)

    def combined_row_mle(self, r_x: Array, r_batch: Array) -> Array:
        """`M(y) = Σ_i eq(r_x)_i · (A + r·B + r²·C)_{i,y}`, length `num_cols`.

        The inner-sumcheck operand: the three matrices batched by powers of `r`,
        then bound on the row variables at `r_x`. MSB-first row order matches the
        outer sumcheck's bind and `expand_eq_to_hypercube`.
        """
        combined = self.a + r_batch * self.b + r_batch * r_batch * self.c
        eq_rows = expand_eq_to_hypercube(r_x, jnp.ones((), self.a.dtype))
        return eq_rows @ combined

    def eval_combined_matrix(self, r_x: Array, r_y: Array, r_batch: Array) -> Array:
        """`Ã(r_x,r_y) + r·B̃ + r²·C̃` as `eq(r_x)·M·eq(r_y)` — the verifier's
        `eval_ABC`. Dense here; a succinct scheme opens it from a SPARK commitment.
        """
        combined = self.a + r_batch * self.b + r_batch * r_batch * self.c
        eq_rows = expand_eq_to_hypercube(r_x, jnp.ones((), self.a.dtype))
        eq_cols = expand_eq_to_hypercube(r_y, jnp.ones((), self.a.dtype))
        return eq_rows @ combined @ eq_cols


def assignment(witness: Array, io: Array, num_vars_padded: int, num_io: int) -> Array:
    """Assemble `z = (W, 1, X)` from the witness and public inputs.

    `W` is padded into the low half `[0, num_vars_padded)`; the high half holds
    the constant `1` at its first slot, then the public inputs `X`, then zero
    padding. Length is `2·num_vars_padded`.
    """
    if witness.shape[0] > num_vars_padded:
        raise ValueError("witness longer than the padded low half")
    if io.shape[0] != num_io:
        raise ValueError(f"expected {num_io} public inputs, got {io.shape[0]}")
    dtype = witness.dtype
    low = jnp.zeros((num_vars_padded,), dtype).at[: witness.shape[0]].set(witness)
    high = jnp.zeros((num_vars_padded,), dtype).at[0].set(jnp.ones((), dtype))
    high = high.at[1 : 1 + num_io].set(io)
    return jnp.concatenate([low, high])


def eval_public_half(io: Array, r_y_rest: Array, num_vars_padded: int) -> Array:
    """MLE of the public high half `(1, X, 0…)` at `r_y[1:]`.

    The verifier evaluates the `(1, X)` part of `z̃` itself; combined with the
    witness opening `eval_W` it reconstructs `z̃(r_y)` — see `pcs_glue`.
    """
    dtype = io.dtype
    high = jnp.zeros((num_vars_padded,), dtype).at[0].set(jnp.ones((), dtype))
    high = high.at[1 : 1 + io.shape[0]].set(io)
    return (high * expand_eq_to_hypercube(r_y_rest, jnp.ones((), dtype))).sum()


def recombine_z_eval(eval_w: Array, eval_pub: Array, r_y0: Array) -> Array:
    """`z̃(r_y)` from the two half-openings: `(1−r_y0)·eval_W + r_y0·eval_pub`.

    `r_y0` (the MSB inner challenge) selects between the witness low half and the
    public high half; the multilinear bind on that top variable is this affine
    combination.
    """
    one = jnp.ones((), eval_w.dtype)
    return (one - r_y0) * eval_w + r_y0 * eval_pub


__all__ = [
    "R1CS",
    "assignment",
    "eval_public_half",
    "recombine_z_eval",
    "eval_eq",
]
