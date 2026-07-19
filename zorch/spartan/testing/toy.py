# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""A satisfying toy R1CS fixture for the Spartan combinator tests.

Builds a dense `(A, B, C)` and a witness-first assignment `z = (W, 1, X)` that
satisfies `(A·z)∘(B·z) = C·z` **by construction**: pick `A, B, W, X` freely, set
`t = (A·z)∘(B·z)`, then place `t` in `C`'s constant-`1` column so `C·z = t`. Any
`A, B, z` yields a satisfied instance, so the fixture stresses the protocol
plumbing, not a specific circuit.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.spartan.r1cs import R1CS, assignment
from zorch.testkit.random_field import rand_field


def toy_r1cs(
    seed: int, s_x: int, num_vars_padded: int, num_io: int, dtype: Any
) -> tuple[R1CS, Array, Array, Array]:
    """Return `(instance, z, witness, io)` for a satisfying dense R1CS.

    `s_x` sets the row count `2^{s_x}`; `num_vars_padded` the witness low-half
    size (so `num_cols = 2·num_vars_padded`, `s_y = log2(num_cols)`).
    """
    num_cons = 1 << s_x
    n = 2 * num_vars_padded
    a = rand_field(seed, (num_cons, n), dtype)
    b = rand_field(seed + 1, (num_cons, n), dtype)
    witness = rand_field(seed + 2, (num_vars_padded,), dtype)
    io = rand_field(seed + 3, (num_io,), dtype)
    z = assignment(witness, io, num_vars_padded, num_io)
    t = (a @ z) * (b @ z)
    const_col = num_vars_padded  # z[const_col] == 1
    c = fnp.zeros((num_cons, n), dtype).at[:, const_col].set(t)
    return R1CS(a=a, b=b, c=c, num_io=num_io), z, witness, io
