# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-round claim reductions, one per sumcheck wire form.

A round's reduction is protocol arithmetic, not verifier machinery: the prover
binds its own challenge into the same reduced claim the verifier will derive, so
both roles call the same function here. Holding it on the verifier round instead
would force a prover that accumulates to either import the verifier or restate
the arithmetic, and a restatement is free to drift.

Each takes the running claim's value, the round message, and the sampled
challenge, and returns `(reduced, ok)`. `ok` is the round's own consistency
verdict — a wire form that spends its redundancy elsewhere returns constant
true, and an honest prover's is true by construction.
"""

from __future__ import annotations

from functools import partial

import frx
import frx.numpy as fnp
from frx import Array

from zorch.poly.univariate import eval_coeffs, eval_univariate
from zorch.round import RunningClaim
from zorch.sumcheck.domain import EvalDomain, subgroup_sum


@partial(frx.jit, static_argnames=("degree",))
def advance_evals(
    claim: RunningClaim, msg: Array, r: Array, degree: int
) -> RunningClaim:
    """Reduce and bind in one compiled step, for the natural-domain wire form.

    The arithmetic is a handful of field ops, but eagerly it is a handful of
    *dispatches* — the Lagrange evaluation, the identity comparison, and the
    scatter `bind` performs each cost more than they compute. A verifier pays
    none of that because its whole replay is one jit zone; a prover folding in a
    host loop has to ask for the same batching explicitly.
    """
    reduced, _ = reduce_evals(claim.value, msg, r, degree)
    return claim.bind(reduced, r)


def require_width(msg: Array, expected: int, kind: str = "evals") -> None:
    """Reject a message whose static shape cannot be this round's polynomial.

    Separate from the reductions so a caller can refuse a malformed proof before
    reading anything off the running claim: a structural error is not a
    soundness failure and must not depend on claim state.
    """
    if msg.shape[0] != expected:
        raise ValueError(
            f"round message must have {expected} {kind}, got {msg.shape[0]}"
        )


def reduce_evals(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """The natural-domain form: `msg` is `s` sampled on `{0..degree}`.

    The redundancy `s(0) + s(1) == claim` is this round's own check.
    """
    require_width(msg, degree + 1)
    return eval_univariate(msg, r), claim == msg[0] + msg[1]


def reduce_coeffs(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """The coefficient form: `s(0) = c_0` and `s(1) = Σc`, read directly."""
    require_width(msg, degree + 1, "coefficients")
    return eval_coeffs(msg, r), claim == msg[0] + fnp.sum(msg)


def reduce_domain(
    claim: Array, msg: Array, r: Array, domain: EvalDomain
) -> tuple[Array, Array]:
    """Any sampling domain: `msg` is `s` sampled at `domain`\'s nodes.

    Generalizes `reduce_evals`, which assumes the naturals `{0..degree}` and so
    is wrong for a round configured with another domain — the compressed Û the
    √-space engine samples at, for one. Going through the domain\'s own
    value→coefficient map makes the identity check read off `s(0) = c_0` and
    `s(1) = Σc` for every domain alike.
    """
    coeffs = domain.to_coeffs(msg)
    return eval_coeffs(coeffs, r), claim == coeffs[0] + fnp.sum(coeffs)


def reduce_compressed(claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
    """The compressed degree-2 form, whose message is `[c_0, c_2]`.

    The linear coefficient never rides the wire: `s(1) = claim - s(0)` with
    `s(0) = c_0`, so `c_1 = s(1) - c_0 - c_2`. That reconstruction consumes the
    `s(0) + s(1) == claim` identity, leaving no per-round redundancy to check —
    `ok` is constant true, and binding rests on the terminal claim check. That is
    the trade the compressed form makes for wire size.
    """
    require_width(msg, 2, "coefficients [c_0, c_2]")
    c0, c2 = msg[0], msg[1]
    c1 = claim - c0 - c0 - c2
    return eval_coeffs(fnp.stack([c0, c1, c2]), r), fnp.bool_(True)


def reduce_subgroup(
    claim: Array, msg: Array, r: Array, skip_rounds: int, degree: int
) -> tuple[Array, Array]:
    """The univariate skip's round 0: `s_0` in ascending-coefficient form.

    The subgroup sibling of `reduce_coeffs` — it swaps the hypercube identity for
    the subgroup sum `claim == Σ_{z∈D} s_0(z)`, which `subgroup_sum` reads off the
    coefficients at multiples of `|D| = 2^skip_rounds`.
    """
    require_width(msg, degree * ((1 << skip_rounds) - 1) + 1, "coefficients")
    return eval_coeffs(msg, r), claim == subgroup_sum(msg, skip_rounds)
