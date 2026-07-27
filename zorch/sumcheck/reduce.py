# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-round claim reductions, one per sumcheck wire form.

Protocol arithmetic, not verifier machinery: the prover binds its own challenge
into the same reduced claim the verifier derives, so both roles call these. Each
returns `(reduced, ok)`, `ok` being the round's own consistency check.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from zorch.poly.univariate import eval_coeffs, eval_univariate
from zorch.sumcheck.domain import EvalDomain, subgroup_sum


def require_width(msg: Array, expected: int, kind: str = "evals") -> None:
    """Reject a malformed message before anything reads the claim: a structural
    error is not a soundness failure and must not depend on claim state."""
    if msg.shape[0] != expected:
        raise ValueError(
            f"round message must have {expected} {kind}, got {msg.shape[0]}"
        )


def reduce_evals(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """`msg` is `s` sampled on the naturals `{0..degree}`."""
    require_width(msg, degree + 1)
    return eval_univariate(msg, r), claim == msg[0] + msg[1]


def reduce_coeffs(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """`msg` is `s`'s coefficients, so `s(0) = c_0` and `s(1) = Σc` read off."""
    require_width(msg, degree + 1, "coefficients")
    return eval_coeffs(msg, r), claim == msg[0] + fnp.sum(msg)


def reduce_domain(
    claim: Array, msg: Array, r: Array, domain: EvalDomain
) -> tuple[Array, Array]:
    """`msg` is `s` sampled at `domain`'s nodes.

    `reduce_evals` assumes the naturals, so it is wrong for a round configured
    with another domain — the √-space engine's compressed Û, for one.
    """
    coeffs = domain.to_coeffs(msg)
    return eval_coeffs(coeffs, r), claim == coeffs[0] + fnp.sum(coeffs)


def reduce_compressed(claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
    """`msg` is `[c_0, c_2]`; `c_1` is reconstructed from `s(1) = claim - c_0`.

    That reconstruction spends the `s(0) + s(1) == claim` identity, so `ok` is
    constant true and binding rests on the terminal check — the trade this form
    makes for wire size.
    """
    require_width(msg, 2, "coefficients [c_0, c_2]")
    c0, c2 = msg[0], msg[1]
    c1 = claim - c0 - c0 - c2
    return eval_coeffs(fnp.stack([c0, c1, c2]), r), fnp.bool_(True)


def reduce_subgroup(
    claim: Array, msg: Array, r: Array, skip_rounds: int, degree: int
) -> tuple[Array, Array]:
    """The skip's round 0: `s_0` in coefficients, checked against the subgroup
    sum `claim == Σ_{z∈D} s_0(z)` rather than the hypercube identity."""
    require_width(msg, degree * ((1 << skip_rounds) - 1) + 1, "coefficients")
    return eval_coeffs(msg, r), claim == subgroup_sum(msg, skip_rounds)
