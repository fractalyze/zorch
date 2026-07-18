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

Committed columns arrive **split by field** — ``base_cols`` then ``cubic_cols``,
in batching order — so a base column keeps its native-width reads instead of
being embedded to the extension up front (one matrix carries one dtype, so a
unified column block would force the embedding, ~one machine word per element
becoming the extension's full width). The composition promotes base against the
extension challenge as it accumulates.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as jnp
from frx import Array

from zorch.poly.univariate import powers


def deep_composition(
    base_cols: Array,
    cubic_cols: Array,
    evals: Array,
    xis: Array,
    opening_pos: Sequence[int],
    vf: Array,
    domain: Array,
) -> Array:
    """``f(x) = Σ_m vf^m · (col_m(x) − evals[m]) / (domain − xis[opening_pos[m]])``
    on ``domain`` — the DEEP-ALI batched quotient.

    Columns arrive split: ``base_cols`` ``(N, B)`` then ``cubic_cols`` ``(N, C)``,
    in batching order, so ``m < B`` indexes a base column and ``m ≥ B`` a cubic
    one (``evals``/``xis``/``vf`` are the extension). ``opening_pos[m]`` selects
    column ``m``'s point. Returns the ``(N,)`` codeword.

    ``M = B + C`` is static, so the loop unrolls: each column's ``vf^m·(col − eval)``
    numerator accumulates into a per-opening-point running sum, then one reciprocal
    per distinct point divides — a fused elementwise graph by construction, no
    ``(N, M)`` intermediate and no ``axis`` reduction. Field addition is exact, so
    the accumulation order does not affect the result."""
    b, c = base_cols.shape[1], cubic_cols.shape[1]
    m = b + c
    vf_pows = powers(vf, m)
    numer_by_opening: dict[int, Array] = {}
    for col in range(m):
        column = base_cols[:, col] if col < b else cubic_cols[:, col - b]
        term = vf_pows[col] * (column - evals[col])  # (N,); base − cubic promotes
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
    cubic_cols: Array,
    weights: Array,
    opening_pos: Sequence[int],
    stride: int = 1,
) -> Array:
    """Evaluate committed columns at their opening points via a precomputed
    barycentric weight matrix — the eval-form source of ``deep_composition``'s
    per-column ``evals``.

    Columns arrive split like ``deep_composition``'s: ``base_cols`` ``(N·stride, B)``
    then ``cubic_cols`` ``(N·stride, C)``, subsampled by ``stride`` to the ``(N,)``
    domain ``weights`` is defined over (``stride = 1`` when already on it).
    ``weights`` is ``(N, K)``: column ``k`` is the Lagrange-evaluation vector at
    the ``k``-th opening point (e.g. ``poly.univariate.compute_lagrange_basis``);
    ``opening_pos[m]`` picks column ``m``'s weights. Returns the ``(M,)`` openings,
    base first.

    ``M`` is static, so each column's dot uses a per-column static weight slice —
    no ``(N, M)`` gather — and a base column's dot keeps its native-width reads."""
    b, c = base_cols.shape[1], cubic_cols.shape[1]
    base_sub = base_cols[::stride]
    cubic_sub = cubic_cols[::stride]
    opened: list[Array] = []
    for col in range(b + c):
        column = base_sub[:, col] if col < b else cubic_sub[:, col - b]
        opened.append(jnp.sum(weights[:, opening_pos[col]] * column))
    return jnp.stack(opened)
