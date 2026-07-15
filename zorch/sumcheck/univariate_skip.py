# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Univariate skip (Gruen's trick, SWIRL-generalized): collapse the first
`skip_rounds` sumcheck rounds into ONE univariate round over a multiplicative subgroup
D of order 2^skip_rounds.

The standard multilinear sumcheck over H_{skip_rounds+n} runs skip_rounds+n rounds,
each binding one boolean variable in the extension field. The skip instead runs the
sumcheck over the hyperprism D × H_n: round 0 is a single univariate round over D — the
boolean prefix {0,1}^skip_rounds is identified index-for-index with the |D| subgroup
points, so each factor's prefix values are the values of a degree-<|D| univariate on D.
Round 0 sends s₀(Z) = Σ_{x∈H_n} combine(factors)(Z, x) in ascending-coefficient form;
the verifier checks c == Σ_{z∈D} s₀(z) (domain.subgroup_sum), samples r₀ ∈ F_ext, and
the claim reduces to s₀(r₀). Round and challenge count drop from skip_rounds+n to 1+n.

Round 0 is base-field work: the factors are base-field, and their D-coefficients come
out of an iNTT (domain.subgroup_to_coeffs), the low-degree extension onto the superset
the degree-`degree·(|D|−1)` message needs comes out of an NTT (domain.subgroup_evals),
and s₀'s coefficients come out of a final iNTT — extension arithmetic starts only once
r₀ is bound at round 1. The 1..n tail is the ordinary eval-form sumcheck over the bound
extension-field state: the very same `StandardRound`, given `ext_dtype` so it squeezes
the extension challenge it folds at (verifier dual verifier.SumcheckRound); any engine
built on StandardRound (e.g. `sqrt_space.prove_sqrt_space`) serves as the tail too.

`skip_rounds == 0` is a strict opt-in off switch: it delegates to the plain
StandardRound run (verifier dual verifier.SumcheckRound), byte-identical to a sumcheck
that never knew about the feature.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import frx
import frx.numpy as jnp
from frx import Array

from zorch.poly.univariate import eval_coeffs
from zorch.prove import fold_rounds
from zorch.sumcheck.domain import subgroup_evals, subgroup_to_coeffs
from zorch.sumcheck.prover import (
    ProductSummand,
    StandardRound,
    SumcheckSummand,
    challenge_limbs,
)
from zorch.sumcheck.verifier import SumcheckRound, UnivariateSkipRound
from zorch.transcript import Transcript, sample_challenge
from zorch.utils.bits import log2_strict_usize


def _next_power_of_two(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


@partial(frx.jit, static_argnums=(1, 2))
def round0_message(
    p_initial: Array, skip_rounds: int, summand: SumcheckSummand
) -> tuple[Array, Array]:
    """The round-0 univariate message s₀ (ascending coefficients, degree
    `degree·(|D|−1)`) and the factors' D-coefficients used to bind r₀.

    Identify the `skip_rounds` most-significant boolean variables with the
    |D| = 2^skip_rounds subgroup points (MSB-first, matching StandardRound's fold):
    reshape each factor to (|D|, H_n) and iNTT the D axis to its degree-<|D|
    Z-coefficients. s₀(Z) = Σ_{x∈H_n} combine(factors)(Z, x); its degree outgrows |D|,
    so the factors are low-degree-extended onto the order-M superset (M the next power
    of two past the degree) before combining, then s₀'s values there are iNTT'd back to
    coefficients. All base-field.

    Jitted (`skip_rounds`/`summand` static) so the NTTs + combine + Σ fuse into one
    dispatch instead of a per-op eager chain."""
    m = p_initial.shape[0]
    size = 1 << skip_rounds
    hn = p_initial.shape[1] // size
    # (m, |D|, H_n) -> D last so the NTT axis is the last one.
    factors = jnp.swapaxes(jnp.reshape(p_initial, (m, size, hn)), 1, 2)
    coeffs_z = subgroup_to_coeffs(factors)  # (m, H_n, |D|) ascending Z-coeffs

    d0 = summand.degree * (size - 1)
    big = _next_power_of_two(d0 + 1)
    lde = subgroup_evals(coeffs_z, big)  # (m, H_n, M) factor values on D_M
    # The scalar-explicit combine (not `_combine`): every SumcheckSummand implements
    # `combine` + `combine_scalars` (product's are empty, LogUp's carry λ), but only
    # ProductSummand also exposes `_combine`, so this stays summand-generic.
    combined = summand.combine(summand.combine_scalars(), *lde)  # (H_n, M)
    s0_vals = jnp.sum(combined, axis=0)  # (M,) s₀ on D_M
    return subgroup_to_coeffs(s0_vals)[: d0 + 1], coeffs_z


def skip_round0(
    p_initial: Array,
    skip_rounds: int,
    transcript: Transcript,
    summand: SumcheckSummand,
    ext_dtype: Any,
) -> tuple[Array, Transcript, Array]:
    """Round 0 of the univariate skip: emit s₀, observe it, sample r₀ ∈ F_ext, and bind
    the prism at r₀. Returns the bound extension MLE state (m, H_n), the transcript, and
    the round-0 message. The caller runs any n-round sumcheck tail over the state —
    `StandardRound` (`prove_univariate_skip`) or a memory-optimized engine
    (`sqrt_space.prove_sqrt_space`) — so the skip stacks with the other round-cost
    levers."""
    limbs = challenge_limbs(ext_dtype)
    msg0, coeffs_z = round0_message(p_initial, skip_rounds, summand)
    transcript = transcript.observe(msg0)
    transcript, r0 = sample_challenge(transcript, ext_dtype, limbs)
    # Bind the prism at r₀: evaluate each factor's D-univariate there. Base coeffs ×
    # extension r₀ promote to the extension the tail runs in.
    return eval_coeffs(coeffs_z, r0), transcript, msg0


def prove_univariate_skip(
    p_initial: Array,
    skip_rounds: int,
    transcript: Transcript,
    summand: SumcheckSummand | None = None,
    ext_dtype: Any | None = None,
) -> tuple[Array, Transcript, list[Array]]:
    """Prove the sumcheck with the first `skip_rounds` rounds collapsed into one
    univariate round over the order-2^skip_rounds subgroup. `summand` defaults to the
    product over the factors; `ext_dtype` is the extension the tail challenges live in
    (defaults to the factor field — a base-field run). Returns the final folded factors
    (m, 1), the transcript, and all 1+n round messages (the round-0 coefficient message
    first).

    `skip_rounds == 0` delegates to the plain StandardRound run — byte-identical to a
    sumcheck without the skip."""
    summand = summand or ProductSummand(degree=p_initial.shape[0])
    total = log2_strict_usize(p_initial.shape[1])
    if not 0 <= skip_rounds <= total:
        raise ValueError(f"skip_rounds must be in [0, {total}], got {skip_rounds}")
    if skip_rounds == 0:
        return fold_rounds(StandardRound(summand), p_initial, transcript, total)

    ext_dtype = ext_dtype or p_initial.dtype
    state, transcript, msg0 = skip_round0(
        p_initial, skip_rounds, transcript, summand, ext_dtype
    )
    state, transcript, tail = fold_rounds(
        StandardRound(summand, ext_dtype=ext_dtype),
        state,
        transcript,
        total - skip_rounds,
    )
    return state, transcript, [msg0] + tail


def verify_univariate_skip(
    claim: Array,
    msgs: list[Array],
    skip_rounds: int,
    total: int,
    transcript: Transcript,
    degree: int,
    ext_dtype: Any | None = None,
) -> tuple[Array, Transcript, list[Array], Array]:
    """Replay the skip prover: the subgroup round-0 check then the coefficient tail,
    threading the claim and ANDing every round's `ok`. Returns the reduced final claim
    (which a consumer checks equals combine(factors)(point)), the transcript, the bound
    point [r₀, …, r_n], and `ok`. `skip_rounds == 0` replays the plain StandardRound run
    (`SumcheckRound`), the exact dual of the prover's off-switch."""
    if skip_rounds == 0:
        rnd = SumcheckRound(degree=degree)
        point: list[Array] = []
        ok = jnp.bool_(True)
        for msg in msgs:
            claim, transcript, r, ok_r = rnd(claim, msg, transcript)
            point.append(r)
            ok = ok & ok_r
        return claim, transcript, point, ok

    ext_dtype = ext_dtype or claim.dtype
    limbs = challenge_limbs(ext_dtype)
    n = total - skip_rounds
    if len(msgs) != 1 + n:
        raise ValueError(f"need {1 + n} messages, got {len(msgs)}")

    # Round 0: the subgroup-sum check + s₀(r₀) reduction.
    claim, transcript, r0, ok = UnivariateSkipRound(
        skip_rounds=skip_rounds,
        degree=degree,
        ext_dtype=ext_dtype,
        challenge_limbs=limbs,
    )(claim, msgs[0], transcript)
    point = [r0]

    # Rounds 1..n: the eval-form tail. Reuse SumcheckRound's identity math
    # (`check_reduce`) but squeeze the extension challenge the tail folds at.
    tail = SumcheckRound(degree=degree)
    for msg in msgs[1:]:
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, ext_dtype, limbs)
        claim, ok_r = tail.check_reduce(claim, msg, r)
        point.append(r)
        ok = ok & ok_r
    return claim, transcript, point, ok
