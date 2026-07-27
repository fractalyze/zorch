# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck replay driver — the dual of `prove`.

The per-round check and claim-reduction live on the verifier round; this driver
only scans it over the proof, threads the transcript, and ANDs the consistency
flags. It stops at the reduced point-claim — the final
`final_claim == oracle(point)` check needs a PCS opening and is the consumer's,
keeping this block proving-scheme- and PCS-agnostic.

The bound point rides the carry in a preallocated buffer rather than the scan's
per-step output channel, so a round reports only its verdict and the round
protocol stays the same shape as one that exports nothing (a GKR layer). The
challenge is derived state, not a message, and derived state belongs in the
carry.

The replay is one `lax.scan` over the proof rows, not a Python loop: the whole
verification compiles to a single traced region that is flat in the round count, so it
stays one fused unit rather than an unrolled body that
crosses the XLA PTX cliff.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

import frx.numpy as fnp
from frx import Array, lax

from zorch.round import RunningClaim
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.round import VerifierRound

_StepOut: TypeAlias = "tuple[tuple[RunningClaim, Transcript], Array]"
_Step: TypeAlias = "Callable[[tuple[RunningClaim, Transcript], Array], _StepOut]"


# Memoized per round so the body handed to `lax.scan` keeps one identity.
# `lax.scan` keys its trace cache on that identity and does not see through a
# `functools.partial` the way `jit` does, so a body defined inside `verify` is a
# fresh object per call that re-traces an identical graph — measured at ~400x
# the cost of the replay it performs. Rounds are frozen dataclasses of static
# config, so a freshly built equal round hits this cache too.
_SCAN_STEPS: dict[Any, _Step] = {}


def _scan_step(verifier: VerifierRound[RunningClaim, Array]) -> _Step:
    """The scan body for `verifier`, built once."""
    cached = _SCAN_STEPS.get(verifier)
    if cached is not None:
        return cached

    def step(carry: tuple[RunningClaim, Transcript], msg: Array) -> _StepOut:
        state, transcript = carry
        state, transcript, ok = verifier(state, transcript, msg)
        return (state, transcript), ok

    _SCAN_STEPS[verifier] = step
    return step


def verify(
    verifier: VerifierRound[RunningClaim, Array],
    claim: Array,
    proof: Array,
    transcript: Transcript,
) -> tuple[Array, Array, Transcript, Array]:
    """Replay `proof` against `claim` → `(point, final_claim, transcript, ok)`.

    `ok` ANDs every round's check; one false anywhere rejects the proof.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")

    init = RunningClaim(claim, fnp.zeros((proof.shape[0],), claim.dtype), fnp.int32(0))
    (state, transcript), oks = lax.scan(_scan_step(verifier), (init, transcript), proof)
    return state.point, state.value, transcript, fnp.all(oks)
