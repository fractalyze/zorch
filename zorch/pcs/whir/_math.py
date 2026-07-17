# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR-specific math shared by the prover and verifier.

Field-representation-safe by construction: every domain point comes from
`ReedSolomon.domain()` (actual field values), so the binary k-fold never needs the
subgroup generator as a host int — there is no Montgomery/canonical conversion to
get wrong. (Generic polynomial evaluation lives in `zorch.poly`; only the pieces
unique to WHIR's coset fold and per-round geometry live here.)
"""

from __future__ import annotations

import frx.numpy as jnp
from frx import Array, lax
from frx.typing import DTypeLike

from zorch.coding.reed_solomon import ReedSolomon, fri_fold_values
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import powers
from zorch.transcript import Transcript


def sample_query_positions(
    transcript: Transcript, stride: int, count: int, dtype: DTypeLike
) -> tuple[Transcript, Array]:
    """Sample `count` WHIR query coset indices in `[0, stride)` — `count`
    consecutive squeezes, each reduced on its CANONICAL low limb.

    Still NOT `pcs.fold.sample_positions`: that batched sampler reorders the draws
    and reduces on Montgomery bits, either of which picks different residues and
    desynchronizes the post-query challenge. What WHIR's convention pins is the
    draw ORDER and the canonical reduction, not the number of squeeze calls — and
    `Transcript.sample(count)` is byte-identical to `count` single squeezes (it
    selects the same `_duplexing` results, just reading a rate-block per
    permutation), so the whole draw is one op instead of `count` sequential ones.
    That matters: the per-draw form dispatched `count` transcript squeezes
    host-side, ~68 ms of pure launch latency at WHIR's query counts. `stride` is a
    power of two, so the canonical reduction equals the low-bit mask. `dtype` is
    the base field the transcript squeezes (positions index a base-field
    codeword)."""
    if count == 0:
        return transcript, jnp.empty((0,), jnp.int32)
    transcript, raw = transcript.sample(count)
    # Per draw, take the low limb exactly as the per-draw form's `[0]` did.
    canonical = lax.bitcast_convert_type(raw, dtype).astype(jnp.uint32)
    canonical = canonical.reshape(count, -1)[:, 0]
    return transcript, (canonical % stride).astype(jnp.int32)


def round_code(
    code: ReedSolomon, r: int, k: int, *, rate_increase: bool = False
) -> ReedSolomon:
    """The RS code the WHIR codeword lives in at round `r` (round 0 is `code`).

    The message length always shrinks by `2^(r·k)` — the `r·k` sumcheck folds
    consumed by round `r`. The block length follows the chosen domain schedule:
    `2^(r·k)` smaller in the constant-rate schedule (blowup fixed), or only `2^r`
    smaller in the rate-increasing schedule (`log_rs -= 1` per round, decoupled
    from `k`, the openvm-stark-backend / SWIRL schedule whose rate climbs each
    round). The two coincide at `k == 1`. WHIR round `r` queries this code; the
    re-encode at the end of round `r` produces `round_code(code, r+1, k, ...)`."""
    message_len = code.message_len >> (r * k)
    block_len = code.block_len >> (r if rate_increase else r * k)
    return ReedSolomon(
        message_len=message_len,
        blowup=block_len // message_len,
        dtype=code.dtype,
        coset_shift=code.coset_shift,
    )


def interp_quadratic_012(e0: Array, e1: Array, e2: Array, x: Array) -> Array:
    """Evaluate at `x` the degree-2 polynomial through (0,e0), (1,e1), (2,e2) — the
    sumcheck claim reduction (the round poly is a product of two multilinears).

    Pure field arithmetic rather than `poly.univariate.eval_univariate`: that
    helper's nested `compute_lagrange_basis` `@jit` + `jnp.dot` mis-lowers when
    inlined under the verifier's own `@jit` zone on frx (eager-only).
    `eval_coeffs` (a single jitted power-chain) composes fine and is reused for the
    coefficient-form evaluations."""
    one = jnp.ones((), x.dtype)
    two = one + one
    s1 = e1 - e0
    s2 = e2 - e1
    p = (s2 - s1) / two
    q = s1 - p
    return (p * x + q) * x + e0


def pow2_powers(z: Array, n: int) -> list[Array]:
    """`[z, z², z⁴, …]` of length `n` — the powers-of-two point a multilinear's
    coefficient vector is evaluated on to realize a univariate evaluation at `z`."""
    out = [z]
    for _ in range(n - 1):
        out.append(out[-1] * out[-1])
    return out


def query_gamma_powers(gamma: Array, count: int) -> Array:
    """`[γ², γ³, …, γ^{count+1}]` — the per-query batching weights (the OOD term
    takes `γ¹`, so queries start at `γ²`), one stacked vector so a round's query
    contributions reduce in a single `(weights · terms).sum()`. `count` (the
    query repetitions) is static and `>= 1`; drop the `[1, γ]` prefix of the
    ascending power vector."""
    return powers(gamma, count + 2)[2:]


def eq_table(point: list[Array]) -> Array:
    """`eq(point, ·)` over the hypercube, indexed to match the weight table the
    sumcheck folds (LSB-first in `point`, mirroring the `[0::2]/[1::2]` fold
    order)."""
    one = jnp.ones((), point[0].dtype)
    return expand_eq_to_hypercube(jnp.stack(point[::-1]), one)


def binary_k_fold(values: Array, alphas: list[Array], coset_points: Array) -> Array:
    """Fold a query coset to the folded polynomial's value.

    `values` are the `2^k` opened codeword values on the coset
    `coset_points = {x·ω_k^j}` (conjugates a half apart: `coset_points[j+h] =
    −coset_points[j]`). `k` successive FRI 2-folds at `alphas` collapse them to the
    folded poly evaluated at `x^{2^k}`. Each step squares the surviving points,
    walking down the squared domains. Single-coset; `vmap` over the query axis."""
    vals = values
    pts = coset_points
    for alpha in alphas:
        h = vals.shape[0] // 2
        vals = fri_fold_values(vals[:h], vals[h:], alpha, pts[:h])
        pts = pts[:h] * pts[:h]
    return vals[0]
