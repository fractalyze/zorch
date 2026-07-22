# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Univariate polynomial evaluation, in evaluation and coefficient form."""

from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
from frx import Array

from zorch.utils.bits import is_power_of_two, log2_strict_usize
from zorch.utils.field import base_field, naturals


def powers(x: Array, n: int) -> Array:
    """``(1, x, x², …, x^{n-1})`` ascending, length ``n`` (``n`` static).

    Built by log-doubling (``powers[m:2m] = powers[:m]·xᵐ``) so the traced graph
    is O(log n), not ``n`` unrolled multiplies — a linear power chain makes the
    fused kernel's operand count scale with ``n`` and past a few thousand entries
    overruns the GPU's kernel-parameter space (the same cliff ``eval_coeffs``
    avoids). The monomial-basis evaluation vector: ``⟨coeffs, powers(x, n)⟩ =
    Σ cᵢ xⁱ``."""
    if n < 1:
        raise ValueError(f"powers needs n >= 1, got {n}")
    out = fnp.ones((1,), dtype=x.dtype)
    step = x
    while out.shape[0] < n:
        out = fnp.concatenate([out, out * step])
        step = step * step
    return out[:n]


def eval_univariate(evals: Array, x: Array) -> Array:
    """Evaluate a univariate given by its values on ``[0, 1, ..., len-1]`` at
    ``x``, by Lagrange interpolation over that integer domain.

    A composer over the jitted basis kernel, so itself un-jitted."""
    nodes = naturals(evals.shape[0], evals.dtype)
    return fnp.dot(evals, compute_lagrange_basis(x, nodes))


def _lagrange_denominators(domain: Array) -> Array:
    """``prod_{j != k} (x_k - x_j)`` per node: exclude-self products via a
    masked matrix whose diagonal holds a typed one (a bare literal is not
    converted to a field dtype inside jit)."""
    n = domain.shape[0]
    one = fnp.ones((), domain.dtype)
    mask = fnp.eye(n, dtype=bool)
    return fnp.prod(fnp.where(mask, one, domain[:, None] - domain[None, :]), axis=1)


@frx.jit
def compute_lagrange_basis(r: Array, domain: Array) -> Array:
    """All Lagrange basis evaluations ``L_{D,k}(r)`` over ``domain``:
    ``L_{D,k}(r) = prod_{j != k} (r - x_j) / (x_k - x_j)``.

    Direct form, not barycentric — barycentric divides by ``(r - node)``,
    which an ``r`` landing on a node would zero."""
    one = fnp.ones((), r.dtype)
    mask = fnp.eye(domain.shape[0], dtype=bool)
    numerators = fnp.prod(fnp.where(mask, one, (r - domain)[None, :]), axis=1)
    return numerators / _lagrange_denominators(domain)


def compute_inv_vandermonde(degree: int, dtype: Any) -> Array:
    """Inverse Vandermonde over the natural domain ``{0..degree}``:
    ``coeffs = M @ evals`` for ``evals[j] = p(j)``.

    Built in the base field — the Lagrange basis for an integer domain lives
    in the prime field — so one matrix serves BF and EF callers; EF
    evaluations promote at multiply time."""
    base = base_field(dtype)
    n = degree + 1
    one = fnp.array(1, base)
    zero = fnp.array(0, base)
    domain = naturals(n, dtype)
    denoms = _lagrange_denominators(domain)
    # Column j = coefficients of L_j(x) = prod_{k != j} (x - k) / denom_j,
    # expanded by repeated (x - k) multiplication over the coefficient list.
    columns = []
    for j in range(n):
        num_coeffs = [one]
        for k in range(n):
            if k != j:
                neg_k = -fnp.array(k, base)
                expanded = [zero] * (len(num_coeffs) + 1)
                for i, c in enumerate(num_coeffs):
                    expanded[i] = expanded[i] + c * neg_k
                    expanded[i + 1] = expanded[i + 1] + c
                num_coeffs = expanded
        columns.append(fnp.stack(num_coeffs) / denoms[j])
    return fnp.stack(columns, axis=1)


# `eval_coeffs` schedule threshold: Horner unroll at or below, prefix-product
# scan above. n = 16 already splits by batch size, so do not raise this without
# re-measuring both regimes (crossover table in PR #456).
_HORNER_MAX_COEFFS = 8


@frx.jit
def eval_coeffs(coeffs: Array, point: Array) -> Array:
    """``p(point) = sum_i coeffs[..., i] * point**i`` — the coefficient-form
    dual of ``eval_univariate``.

    Two schedules, byte-identical (field mul/add are exact and associative, so
    any re-parenthesization is the same element), dispatched on the static
    coefficient count:

    - ``n <= _HORNER_MAX_COEFFS``: an unrolled Horner chain — O(n) multiply-adds
      consuming ``coeffs`` one slice at a time, so it needs no power vector and
      no reduction, and fuses through a producer's pending expression stack
      (e.g. ``intt_with_root``'s in ``fri_fold_k``).
    - larger ``n``: the power vector ``point**i`` as a ``lax.associative_scan``
      (log-depth prefix product), then one batched dot. A *sequential* scan over
      the coefficient axis lowers to a ``while`` the GPU runtime launches
      host-side once per iteration, so an n-coefficient eval pays n launch
      latencies — WHIR's out-of-domain eval at ``n = 2^13`` measured ~550 ms of
      pure dispatch. The prefix product is O(log n) kernels instead, and each
      stage takes one array operand, so the degree never inflates the kernel's
      operand count (an unrolled power chain would, and overflow the GPU's 32 KB
      kernel-parameter space — ``ptxas: too much parameter space``)."""
    n = coeffs.shape[-1]
    if n <= _HORNER_MAX_COEFFS:
        folded = coeffs[..., -1]
        for m in range(n - 2, -1, -1):
            folded = folded * point + coeffs[..., m]
        return folded
    # ``point**i`` as the prefix product of ``[1, point, point, …]`` (n entries),
    # laid out on the last axis so it dots the coefficients' degree axis directly.
    seq = fnp.where(
        fnp.arange(n) == 0,
        fnp.ones_like(point)[..., None],
        point[..., None],
    )
    powers = frx.lax.associative_scan(lambda a, b: a * b, seq, axis=-1)
    return fnp.sum(coeffs * powers, axis=-1)


def ntt_with_root(groups: Array, omega: Array, coset: Array | None = None) -> Array:
    """Evaluate degree-``<k`` coefficients on the order-``k`` subgroup ``⟨ω⟩``
    (``coset=None``) or a coset ``s·⟨ω⟩`` (``coset = s``), for the whole batch at
    once. The exact inverse of ``intt_with_root``: ``groups`` is ``(..., k)``
    coefficients with ``x^m`` at index ``m``, and the result is ``(..., k)``
    values, index ``j`` at ``ωʲ`` (or ``s·ωʲ``).

    Shared-twiddle NTT, mirroring ``intt_with_root``: apply the shift
    ``coeffₘ ·= sᵐ`` (``P(s·y)=Q(y)``), bit-reverse, then ``log k`` butterfly
    stages whose twiddles ``ω⁰..ω^(k/2-1)`` are computed once and shared across
    the batch — no ``k⁻¹``, which is the INTT's alone.

    ``ω`` must have order exactly ``k``: the butterfly rests on ``ω^(k/2) = −1``,
    and a wrong-order root yields silent garbage, not an error. It cannot be
    checked here — ``ω`` is traced, so the test would force a host sync. That is
    the trade the name states: you supply the root, in your own domain order,
    where ``lax.ntt`` derives a canonical one from a generator and length.

    Same regime as ``intt_with_root`` and the same reason: the unrolled stages
    are plain elementwise ops, so adjacent work (a coset shift, an evaluation at
    a point) fuses into one kernel, where ``lax.ntt`` is an opaque kernel
    boundary — measured 3.2x on the k=8 fold (PR #456). For a long transform use
    ``lax.ntt`` (what ``ReedSolomon.encode`` does); this one would unroll ``k``
    stages into the graph.
    """
    k = groups.shape[-1]
    if not is_power_of_two(k):
        raise ValueError(f"ntt_with_root needs a power-of-two factor k, got {k}")
    log_k = log2_strict_usize(k)
    if fnp.ndim(omega) != 0:
        raise ValueError(f"omega must be a scalar, got shape {fnp.shape(omega)}")
    if coset is not None:
        try:
            coset = fnp.broadcast_to(coset, groups.shape[:-1])
        except ValueError as e:
            raise ValueError(
                f"coset shape {fnp.shape(coset)} must broadcast to the "
                f"batch dims {groups.shape[:-1]}"
            ) from e

    col = [groups[..., i] for i in range(k)]

    # Shift first — the mirror of `intt_with_root` undoing it last.
    if coset is not None:
        shift_pow = coset
        for m in range(1, k):
            col[m] = col[m] * shift_pow
            if m < k - 1:
                shift_pow = shift_pow * coset

    twiddles = powers(omega, max(k >> 1, 1))
    col = [col[int(f"{i:0{log_k}b}"[::-1], 2)] for i in range(k)] if k > 1 else col
    half = k >> 1
    for i in range(log_k):
        half_group = 1 << i
        for j in range(half):
            group = j >> i
            offset = j & (half_group - 1)
            i1 = (group << (i + 1)) + offset
            i2 = i1 + half_group
            odd = col[i2] * twiddles[offset * (k >> (i + 1))]
            col[i1], col[i2] = col[i1] + odd, col[i1] - odd

    return fnp.stack(col, axis=-1)


def intt_with_root(
    groups: Array, omega_inv: Array, coset_inv: Array | None = None
) -> Array:
    """Recover the degree-``<k`` coefficients from evaluations on the order-``k``
    subgroup ``⟨ω⟩`` (``coset_inv=None``) or a coset ``s·⟨ω⟩`` (``coset_inv =
    s⁻¹``), for the whole batch at once. ``groups`` is ``(..., k)`` — ``groups[...,
    j]`` is the value at ``ωʲ`` (or ``s·ωʲ``), ``k`` a power of two on the last
    axis; returns ``(..., k)`` coefficients, ``x^m`` at index ``m``.

    Shared-twiddle inverse NTT: bit-reverse, then ``log k`` decimation-in-time
    butterfly stages whose twiddles ``ω⁻⁰..ω⁻^(k/2-1)`` are computed once and
    shared across the whole batch, ``k⁻¹`` normalise, then (coset only) undo the
    shift ``coeffₘ ·= coset_inv^m`` (``Q(y)=P(s·y)``). The coefficient-form
    counterpart of ``compute_lagrange_basis``'s evaluation form; byte-identical
    to ``compute_inv_vandermonde`` or any correct INTT, since field arithmetic is
    exact and the summation order a butterfly vs a matmul takes is irrelevant.

    Carries no root convention (like ``compute_lagrange_basis``): the caller
    supplies ``ω⁻¹`` and per-group ``coset_inv`` as runtime values in its own
    domain order, where ``lax.ntt`` selects its root through a static integer
    ``generator`` (as ``g^((p-1)/n)``). Two primitive roots ``ω`` and ``ωᵗ``
    give the same values under the relabelling ``j ↦ t·j mod k``, so mixing the
    conventions costs a gather (cf. the zk↔pil2 reindex in ZisK's
    ``trace_commit``). ``ω⁻¹`` must have order exactly ``k`` — the butterfly
    rests on ``ω^(k/2) = −1``, and a wrong-order root is silently wrong rather
    than an error, unverifiable here without a host sync on a traced value.

    The butterfly is hand-unrolled over the static ``k`` over a large *batch*
    axis, so it lowers to fused elementwise kernels the per-group shift and a
    downstream evaluation join — ``lax.ntt`` computes the same bytes (verified)
    but as an opaque kernel boundary: the fold through it measures 3.2x slower
    at 2^23/k=8 (PR #456)."""
    k = groups.shape[-1]
    if not is_power_of_two(k):
        raise ValueError(f"intt_with_root needs a power-of-two factor k, got {k}")
    log_k = log2_strict_usize(k)
    if fnp.ndim(omega_inv) != 0:
        raise ValueError(
            f"omega_inv must be a scalar, got shape {fnp.shape(omega_inv)}"
        )
    if coset_inv is not None:
        # `broadcast_to`, not `broadcast_shapes`: the latter is symmetric, so a
        # `coset_inv` of `(3, 1)` against batch dims `(3,)` passes as compatible
        # and then silently widens the result to `(3, 3, k)`, breaking the
        # documented `(..., k)` contract. Broadcasting *to* the batch dims admits
        # only shapes that fit them.
        try:
            coset_inv = fnp.broadcast_to(coset_inv, groups.shape[:-1])
        except ValueError as e:
            raise ValueError(
                f"coset_inv shape {fnp.shape(coset_inv)} must broadcast to the "
                f"batch dims {groups.shape[:-1]}"
            ) from e

    # Shared INTT twiddles ω⁻⁰..ω⁻^(k/2-1), one set for every group. The static k
    # axis becomes a Python list so the butterfly unrolls to elementwise ops over
    # the (batched) leading dims.
    twiddles = powers(omega_inv, max(k >> 1, 1))
    col = [groups[..., i] for i in range(k)]

    # Decimation-in-time INTT: bit-reverse, then log_k butterfly stages.
    col = [col[int(f"{i:0{log_k}b}"[::-1], 2)] for i in range(k)] if k > 1 else col
    half = k >> 1
    for i in range(log_k):
        half_group = 1 << i
        for j in range(half):
            group = j >> i
            offset = j & (half_group - 1)
            i1 = (group << (i + 1)) + offset
            i2 = i1 + half_group
            odd = col[i2] * twiddles[offset * (k >> (i + 1))]
            col[i1], col[i2] = col[i1] + odd, col[i1] - odd

    # Scale by k⁻¹ (INTT normalisation), then (coset only) undo the shift.
    base = base_field(groups.dtype)
    inv_k = fnp.ones((), base) / fnp.asarray(k, base)
    col = [c * inv_k for c in col]
    if coset_inv is not None:
        shift_pow = coset_inv
        for m in range(1, k):
            col[m] = col[m] * shift_pow
            if m < k - 1:  # the final power is never read — skip its elementwise mul
                shift_pow = shift_pow * coset_inv

    return fnp.stack(col, axis=-1)
