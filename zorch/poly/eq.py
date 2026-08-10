# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Equality polynomial eq(w, x) hypercube expansion.

eq(w, x) = Π_i (1 - x_i - w_i + 2·x_i·w_i); Σ_{w∈{0,1}^n} eq(w,x) = 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import frx.numpy as fnp
from frx import Array


def eq_factor(t: Array, z: Array) -> Array:
    """One coordinate's eq factor eq(t, z) = t·z + (1-t)(1-z), elementwise over
    any broadcastable shapes. Symmetric in ``t``/``z``.

    The building block ``eval_eq`` multiplies across coordinates, and the
    per-round factor a sumcheck binds one variable at a time: a prover that
    folds variable z_k at challenge r multiplies its running bound-eq mass by
    ``eq_factor(r, z_k)`` (the jagged provers' ``pad_adj``/``eq_adj``
    accumulation step)."""
    one = fnp.ones((), z.dtype)
    return t * z + (one - t) * (one - z)


def eq_root(z: Array) -> Array:
    """The root (in t) of the eq factor: ``eq_factor(eq_root(z), z) == 0`` at
    ``b = (1-z)/(1-2z)``, elementwise over any shape.

    Both sides derive ``b`` from ``z`` alone, so a round polynomial whose
    summand carries the current variable's eq factor has a known zero there --
    the free interpolation point of the Gruen round-poly compression
    (https://eprint.iacr.org/2024/108, `zorch.sumcheck.gruen`). Undefined at
    z = 1/2 (the factor is constant) and colliding with the t = 1 node at
    z = 0; a transcript-sampled ``z`` avoids both w.h.p."""
    one = fnp.ones((), z.dtype)
    return (one - z) / (one - fnp.array(2, z.dtype) * z)


def eval_eq(w: Array, x: Array) -> Array:
    """eq(w, x) = Π_i (1 - w_i - x_i + 2·w_i·x_i) for two equal-length points, in
    O(len) time and O(1) memory.

    The closed form of the equality polynomial evaluated at a pair of points --
    equals ``(expand_eq_to_hypercube(x, 1) · expand_eq_to_hypercube(w, 1)).sum()``
    but without materializing either 2^len vector, so a verifier evaluating eq at
    a bound point stays succinct. Symmetric in ``w``/``x`` and order-agnostic (a
    product over coordinates), so MSB/LSB indexing does not matter."""
    return fnp.prod(eq_factor(w, x), axis=-1)


def expand_hypercube_step(state: Array, coord: Array, *, msb: bool = False) -> Array:
    """(2ᵏ,) -> (2ᵏ⁺¹,): add a new variable's (1-coord)/coord split — LSB (default)
    interleaves the shares, msb=True concatenates [low, high]."""
    high = state * coord
    low = state - high
    if msb:
        return fnp.concatenate([low, high])
    return fnp.column_stack([low, high]).flatten()


def contract_hypercube_step(state: Array) -> Array:
    """(2^{k+1},) -> (2^k,): Σ-marginalize the LSB variable by summing adjacent
    pairs, ``out[j] = state[2j] + state[2j+1]`` (over the last axis).

    The mass-preserving dual of `expand_hypercube_step`: expand splits each
    entry into ``(1-coord)``/``coord`` shares, so contracting recovers the
    pre-expansion table exactly -- ``Σ_b eq((w, b), x) = eq(w, x[:-1])``. A
    fixed-shape round loop that binds variables LSB-first keeps its eq table
    current with one of these per round instead of re-expanding. The last axis
    must be even."""
    return state[..., 0::2] + state[..., 1::2]


# Above this variable count, the table is built as an outer product of two
# half tables instead of one doubling chain. XLA fuses a chain's tail layers
# into a single concatenate fusion that recomputes each output element's
# ancestor factors (one multiply per fused layer per element), so a long chain
# pays several multiplies per entry and re-materializes every doubling; the
# outer product is exactly one multiply per entry over two small inputs.
_OUTER_SPLIT_MIN = 16


def _flat_outer(slow: Array, fast: Array) -> Array:
    """Flattened outer product; `slow` owns the high index bits."""
    return (slow[:, None] * fast[None, :]).reshape(-1)


def expand_eq_to_hypercube(x: Array, scalar: Array, *, msb: bool = False) -> Array:
    """scalar·eq(w, x) for all w in {0,1}^n. `msb=False` interleaves each new share
    (default, `w[0]` the MSB); `msb=True` concatenates `[low, high]`, placing
    `x[j]` at bit `j`.

    NOTE: explicit indexing instead of `for coord in x` — iterating a JAX array
    of an extension-field dtype dispatches `lax.sign`, a XLA gotcha.
    """
    n = x.shape[0]
    if n >= _OUTER_SPLIT_MIN:
        # out[w] factors over any coordinate split, so the full table is the
        # outer product of the two half tables — with the slow-axis half being
        # whichever slice owns the high index bits (x[:k] when w[0] is the MSB,
        # x[k:] when msb=True places x[j] at bit j). GF multiplication is
        # exact, so the product is byte-equal to the chain.
        k = n // 2
        first = expand_eq_to_hypercube(x[:k], scalar, msb=msb)
        rest = expand_eq_to_hypercube(x[k:], fnp.ones((), x.dtype), msb=msb)
        outer, inner = (rest, first) if msb else (first, rest)
        return _flat_outer(outer, inner)
    state = fnp.atleast_1d(scalar)
    for j in range(n):
        state = expand_hypercube_step(state, x[j], msb=msb)
    return state


@overload
def expand_eq_family(
    cs: Array, *, msb: bool = ..., suffix: bool = ..., keep: None = ...
) -> list[Array]: ...


@overload
def expand_eq_family(
    cs: Array, *, msb: bool = ..., suffix: bool = ..., keep: Callable[[int], bool]
) -> list[Array | None]: ...


def expand_eq_family(
    cs: Array,
    *,
    msb: bool = False,
    suffix: bool = False,
    keep: Callable[[int], bool] | None = None,
) -> list[Array] | list[Array | None]:
    """The nested eq tables [eq(s₁), …, eq(sₙ)] over every prefix s_k = cs[:k]
    (`suffix=True`: every suffix s_k = cs[n-k:]), entry k-1 of shape (2ᵏ,) —
    the prefix family appends each coordinate walking forwards, the suffix
    family prepends walking backwards, and `msb` is the placement of each added
    coordinate as in `expand_hypercube_step`. The slice's first coordinate
    lands at the MSB for the (prefix, msb=False) and (suffix, msb=True)
    combinations, at the LSB for the other two.

    Past `_OUTER_SPLIT_MIN` each large member is emitted as ONE outer product
    of a shared half instead of a retained doubling chain (see
    `_OUTER_SPLIT_MIN` for why): every member factors over the family's
    first-added half — for the suffix family, eq(cs[i:]) = eq(cs[i:k]) ⊗
    eq(cs[k:]) for every i < k — so both halves recurse on half-size families
    and each large table is one write-only GF multiply per element over two
    small inputs. GF multiplication is exact, so every member stays byte-equal
    to its chain.

    keep: optional predicate on the member index — the entries the caller
    will read. Kept entries are always present and exact; un-kept entries
    come back as None, and past the outer-product split their emissions are
    skipped entirely — an un-read half recurses no further. The elision must
    live here in the emitter: families are built eagerly in round
    constructors (`zorch.sumcheck.eq.eq_poly`), where every emission
    materializes, and even under jit an explicit skip keeps the contract
    instead of leaning on compiler DCE. Below the split the chain layers are
    each other's building blocks, so nothing can be skipped — un-kept
    entries are only masked."""
    n = cs.shape[0]
    if keep is None:
        return _expand_family(cs, msb, suffix, [True] * n)
    sel = [keep(i) for i in range(n)]
    family = _expand_family(cs, msb, suffix, sel)
    return [t if s else None for t, s in zip(family, sel, strict=True)]


def _expand_family(
    cs: Array, msb: bool, suffix: bool, sel: list[bool]
) -> list[Array | None]:
    """The recursion behind `expand_eq_family`: selected ⇒ present and exact,
    un-selected ⇒ None — except entries that exist as building blocks anyway
    (chain layers, a forced shared factor), which the public wrapper masks,
    keeping the contract predicate-defined rather than split-regime-defined."""
    n = cs.shape[0]
    if n < _OUTER_SPLIT_MIN:
        state = fnp.ones(1, dtype=cs.dtype)
        tables: list[Array | None] = []
        order = range(n - 1, -1, -1) if suffix else range(n)
        for j in order:
            state = expand_hypercube_step(state, cs[j], msb=msb)
            tables.append(state)
        return tables
    k = n // 2
    # The first-added half (base) is the one every larger member shares. Base
    # members keep their global indices; growth member i grows the product at
    # global index b + i, so the halves split `sel` at b. An entirely un-read
    # growth half recurses no further, and only a read one forces the base's
    # largest member into existence as the shared factor.
    base_cs, growth_cs = (cs[k:], cs[:k]) if suffix else (cs[:k], cs[k:])
    b = n - k if suffix else k
    if not any(sel[b:]):
        return _expand_family(base_cs, msb, suffix, sel[:b]) + [None] * (n - b)
    base_sel = sel[:b]
    base_sel[b - 1] = True
    base = _expand_family(base_cs, msb, suffix, base_sel)
    growth = _expand_family(growth_cs, msb, suffix, sel[b:])
    shared = base[-1]
    assert shared is not None  # forced selected just above
    # `msb` says which index bits the shared half owns (msb=True places
    # first-added coordinates at the low bits — the fast axis — msb=False at
    # the high bits). Selected ⇒ g present; the None check only narrows.
    return base + [
        (
            (_flat_outer(g, shared) if msb else _flat_outer(shared, g))
            if s and g is not None
            else None
        )
        for g, s in zip(growth, sel[b:], strict=True)
    ]


def expand_monomial_step(state: Array, coord: Array) -> Array:
    """(2^k,) -> (2^{k+1},): add a new variable as the LSB, monomial basis.
    result[2j] = state[j], result[2j+1] = state[j]·coord — the ⊗(1, coord)
    factor, where `expand_hypercube_step` is ⊗(1-coord, coord)."""
    return fnp.column_stack([state, state * coord]).flatten()


def expand_monomial_to_hypercube(x: Array, scalar: Array) -> Array:
    """scalar·Π_{i: w_i=1} x_i for all w in {0,1}^n — the monomial
    (coefficient-basis) dual of `expand_eq_to_hypercube`, same (2^n,) shape and
    MSB-first indexing (w[0] binds x[0]). `<coeffs, expand_monomial(x)>` is the
    monomial-basis evaluation at x, as `<evals, expand_eq(x)>` is the eval-basis
    one."""
    state = fnp.atleast_1d(scalar)
    for j in range(x.shape[0]):
        state = expand_monomial_step(state, x[j])
    return state
