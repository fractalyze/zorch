# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`FieldOps` — an explicit field-arithmetic seam for sumchecks over fields that
are NOT a jax-native dtype.

zorch's native sumcheck driver (`sumcheck/prover.py`) does field arithmetic with
the operators `+`/`-`/`*` and `jnp.sum`, which ARE the field ops when the state
is a jax field dtype (a zk_dtypes prime/extension field). A binary field carried
as raw `uint64[..., 2]` lanes — GF(2^128) in a fixed byte basis, where `+` is XOR
and `*` is a carryless GHASH multiply, the representation a prover must use when
byte-identity with an external verifier forbids a native dtype — cannot use those
operators. `FieldOps` abstracts the operations a multilinear sumcheck needs so
such a field can supply its own; `NativeFieldOps` recovers the operator behaviour
as the default, so the seam and the native driver cannot drift on the native path.

This is the seam only. The field-and-wire-generic scan driver that would consume
it (`prove_explicit`, for in-scan Fiat-Shamir — "Path A") is intentionally NOT
built yet: it has no consumer. Binary-field schemes today run their own host
round loop (their Fiat-Shamir is a sequential host hash, see `byte_transcript`)
and reuse just this seam plus their own per-round primitives. Add the scan driver
when a scheme actually adopts in-scan FS.
"""
from __future__ import annotations

from typing import Any, Protocol

import jax.numpy as jnp
from jax import Array


class FieldOps(Protocol):
    """The field arithmetic a multilinear sumcheck round needs, made explicit so a
    non-native-dtype field can supply its own.

    `add`/`sub` coincide in characteristic 2. `sum` reduces an axis under the
    field's addition (XOR-reduce for a binary field). `mul` is the field product.
    `domain_point(u, like)` lifts a small integer `u ∈ {0,1,…,degree}` to a field
    element on a round's evaluation domain (the degree+1-evals message form); a
    field whose round message carries no such domain — e.g. the Karatsuba ∞-trick
    `(G(1), G(∞))` form — raises, signalling that its round owns the message.
    """

    @property
    def zero(self) -> Array: ...
    @property
    def one(self) -> Array: ...
    def add(self, a: Array, b: Array) -> Array: ...
    def sub(self, a: Array, b: Array) -> Array: ...
    def mul(self, a: Array, b: Array) -> Array: ...
    def sum(self, x: Array, *, axis: int) -> Array: ...
    def domain_point(self, u: int, like: Array) -> Array: ...
    def zeros_like(self, x: Array) -> Array: ...


class NativeFieldOps:
    """`FieldOps` for a jax-native field dtype — `+`/`-`/`*`, `jnp.sum`, and
    integer domain points. Reproduces the arithmetic of `sumcheck/prover.py`
    exactly (`fold_pair`, `lift_to_domain`, `_round_poly`), so a round expressed
    over `FieldOps` and instantiated with `NativeFieldOps(dtype)` is byte-identical
    to the native driver on that field."""

    def __init__(self, dtype: Any):
        self._dtype = dtype

    @property
    def zero(self) -> Array:
        return jnp.zeros((), self._dtype)

    @property
    def one(self) -> Array:
        return jnp.ones((), self._dtype)

    def add(self, a: Array, b: Array) -> Array:
        return a + b

    def sub(self, a: Array, b: Array) -> Array:
        return a - b

    def mul(self, a: Array, b: Array) -> Array:
        return a * b

    def sum(self, x: Array, *, axis: int) -> Array:
        return jnp.sum(x, axis=axis)

    def domain_point(self, u: int, like: Array) -> Array:
        # Matches lift_to_domain's `jnp.array(u, p0.dtype)` (prover.py).
        return jnp.array(u, like.dtype)

    def zeros_like(self, x: Array) -> Array:
        return jnp.zeros_like(x)
