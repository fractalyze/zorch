# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-round claim reductions, shared by both roles of a sumcheck round.

A round's claim reduction is protocol arithmetic, not verifier machinery: the
prover binds its own challenge into the same reduced claim the verifier will
derive. Keeping the arithmetic here means one definition per wire form, so the
two roles cannot drift, and neither has to import the other to agree.

Each function takes the running claim value, the round message, and the sampled
challenge, and returns `(reduced, ok)`. `ok` is the round's own consistency
verdict; a prover's is true by construction and it discards it.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from zorch.poly.univariate import eval_coeffs, eval_univariate


def require_width(msg: Array, expected: int, kind: str = "evals") -> None:
    """Reject a message whose static shape cannot be this round's polynomial.

    Separate from the reductions so a caller can reject a malformed proof before
    reading anything off the running claim: a structural error is not a
    soundness failure and must not depend on claim state.
    """
    if msg.shape[0] != expected:
        raise ValueError(
            f"round message must have degree+1={expected} {kind}, "
            f"got {msg.shape[0]}"
        )


def reduce_evals(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """Reduce the natural-domain wire form: `msg` is `s` on `{0..degree}`.

    The redundancy `s(0) + s(1) == claim` is the round's own check.
    """
    require_width(msg, degree + 1)
    return eval_univariate(msg, r), claim == msg[0] + msg[1]


def reduce_coeffs(
    claim: Array, msg: Array, r: Array, degree: int
) -> tuple[Array, Array]:
    """Reduce the coefficient wire form: `msg` is `s`'s coefficients."""
    require_width(msg, degree + 1, "coefficients")
    return eval_coeffs(msg, r), claim == msg[0] + fnp.sum(msg)


def reduce_compressed(claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
    """Reduce the compressed degree-2 form, whose message is `[c_0, c_2]`.

    The linear coefficient never rides the wire: `s(1) = claim - s(0)` with
    `s(0) = c_0`, so `c_1 = s(1) - c_0 - c_2`. That reconstruction consumes the
    `s(0) + s(1) == claim` identity, leaving no per-round redundancy to check —
    `ok` is constant true, and binding rests on the terminal claim check. This
    is the trade the compressed form makes for wire size.
    """
    if msg.shape[0] != 2:
        raise ValueError(
            f"compressed round message must carry [c_0, c_2], got shape {msg.shape}"
        )
    c0, c2 = msg[0], msg[1]
    c1 = claim - c0 - c0 - c2
    return eval_coeffs(fnp.stack([c0, c1, c2]), r), fnp.bool_(True)
