# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""DEEP-ALI composition: batch committed columns' out-of-domain quotients into
one low-degree codeword the fold phase covers.

Scheme-neutral. A DEEP-ALI opening reduces the claim "column ``p_m`` equals
``eval_m`` at the out-of-domain point ``ξ_m``" to the quotient
``(p_m(x) − eval_m)/(x − ξ_m)`` — a genuine polynomial exactly when the claim
holds — and batches the ``M`` of them by powers of one Fiat-Shamir challenge into
a single polynomial the low-degree test covers. The domain, the openings, and the
batching challenge are the caller's; this module owns only the eval-form
arithmetic, so it lives at the ``pcs`` level rather than under any one scheme's
package (cf. ``pcs/fold.py``).

Committed columns arrive **split by field** — ``base_cols`` then ``ext_cols``,
in batching order — so a base column keeps its native-width reads instead of
being embedded to the extension up front (one matrix carries one dtype, so a
unified column block would force the embedding, ~one machine word per element
becoming the extension's full width). The composition promotes base against the
extension challenge as it accumulates.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array

from zorch.poly.univariate import powers


def deep_composition(
    base_cols: Array,
    ext_cols: Array,
    evals: Array,
    xis: Array,
    opening_pos: Sequence[int],
    vf: Array,
    domain: Array,
    vf_pows: Array | None = None,
    columns_leading: bool = False,
) -> Array:
    """``f(x) = Σ_m vf^m · (col_m(x) − evals[m]) / (domain − xis[opening_pos[m]])``
    on ``domain`` — the DEEP-ALI batched quotient.

    Columns arrive split: ``base_cols`` ``(N, B)`` then ``ext_cols`` ``(N, C)``,
    in batching order, so ``m < B`` indexes a base column and ``m ≥ B`` an
    extension one (``evals``/``xis``/``vf`` are the extension).
    ``opening_pos[m]`` selects column ``m``'s point. Returns the ``(N,)`` codeword.

    ``columns_leading=True`` takes the same columns transposed — ``(B, N)`` and
    ``(C, N)`` — which is how a producer that transforms along the last axis
    (an LDE, say) already holds them. Reading a column is then contiguous, so
    consecutive rows land in consecutive addresses and the warp coalesces;
    row-major makes the same reads a full row apart. Prefer this form when the
    caller can supply it, with one exception: at large ``M`` this kernel holds
    every column live at once, and elements-per-thread × live columns is what
    exhausts the register file. The compiler unrolls precisely because these
    reads are contiguous, so the column-major form is where that bites — 2.5×
    at M=68, N=2²². Splitting the batch across several calls and summing the
    partial numerators keeps each kernel narrow enough to avoid it.

    ``M = B + C`` is static, so the loop unrolls: each column's ``vf^m·(col − eval)``
    numerator accumulates into a per-opening-point running sum, then one reciprocal
    per distinct point divides — a fused elementwise graph by construction, no
    ``(N, M)`` intermediate and no ``axis`` reduction. Field addition is exact, so
    the accumulation order does not affect the result.

    ``vf_pows`` (``(M,)``) overrides the default ascending ``vf^m`` when the
    caller fixes a different power-to-column assignment — e.g. descending,
    where column 0 carries the highest power (a Horner-style accumulation
    order). It is also a performance lever, and which way it points depends on
    the layout: derived in-graph the powers are an M-long dependent chain the
    compiler may fold into the per-row body, which costs 3× under
    ``columns_leading`` (M=68, N=2²¹) but is slightly cheaper than a load in
    the row-major form. Materialize them outside the jit whenever
    ``columns_leading`` is set."""
    axis = 0 if columns_leading else 1
    b, c = base_cols.shape[axis], ext_cols.shape[axis]
    m = b + c
    if vf_pows is None:
        vf_pows = powers(vf, m)
    numer_by_opening: dict[int, Array] = {}
    for col in range(m):
        if columns_leading:
            column = base_cols[col] if col < b else ext_cols[col - b]
        else:
            column = base_cols[:, col] if col < b else ext_cols[:, col - b]
        term = vf_pows[col] * (column - evals[col])  # (N,); base − ext promotes
        o = opening_pos[col]
        numer_by_opening[o] = (
            term if o not in numer_by_opening else numer_by_opening[o] + term
        )
    f: Array | None = None
    for o, numer in numer_by_opening.items():
        term = numer / (domain - xis[o])
        f = term if f is None else f + term
    assert f is not None  # M >= 1, so at least one opening group accumulated
    return f


def open_columns(
    base_cols: Array,
    ext_cols: Array,
    weights: Array,
    opening_pos: Sequence[int],
    stride: int = 1,
) -> Array:
    """Evaluate committed columns at their opening points via a precomputed
    barycentric weight matrix — the eval-form source of ``deep_composition``'s
    per-column ``evals``.

    Columns arrive split like ``deep_composition``'s: ``base_cols`` ``(N·stride, B)``
    then ``ext_cols`` ``(N·stride, C)``, subsampled by ``stride`` to the ``(N,)``
    domain ``weights`` is defined over (``stride = 1`` when already on it).
    ``weights`` is ``(N, K)``: column ``k`` is the Lagrange-evaluation vector at
    the ``k``-th opening point (e.g. ``poly.univariate.compute_lagrange_basis``);
    ``opening_pos[m]`` picks column ``m``'s weights. Returns the ``(M,)`` openings,
    base first.

    Each opening is ``Σ_k weights[k]·col[k]`` over the domain. ``opening_pos`` is
    static, so gathering each column's weight vector is a compile-time column
    select; the whole base block then reduces at once and the extension block at
    once — a couple of block dots, not one launch per column. Base and extension
    stay in separate blocks so a base column keeps native-width reads."""
    b = base_cols.shape[1]
    base_sub = base_cols[::stride]
    ext_sub = ext_cols[::stride]
    base_w = weights[:, np.array(opening_pos[:b], dtype=np.intp)]  # (N, B)
    ext_w = weights[:, np.array(opening_pos[b:], dtype=np.intp)]  # (N, C)
    base_openings = fnp.sum(base_w * base_sub, axis=0)
    ext_openings = fnp.sum(ext_w * ext_sub, axis=0)
    return fnp.concatenate([base_openings, ext_openings])
