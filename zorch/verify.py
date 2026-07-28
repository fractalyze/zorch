# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck replay driver — the dual of `prove`.

Stops at the reduced point-claim: the final `final_claim == oracle(point)` check
needs a PCS opening, so it is the consumer's and this block stays scheme- and
PCS-agnostic. The bound point rides the carry rather than a per-step output, so
a round reports only its verdict and every recurrence shape shares one protocol.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
from frx import Array

from zorch.round import RunningClaim
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.round import VerifierRound


@partial(frx.jit, static_argnames=("verifier",))
def verify(
    verifier: VerifierRound[RunningClaim, Array],
    claim: Array,
    proof: Array,
    transcript: Transcript,
) -> tuple[Array, Array, Transcript, Array]:
    """Replay `proof` against `claim` → `(point, final_claim, transcript, ok)`.

    `ok` ANDs every round's check; one false anywhere rejects the proof.

    **Requires a device-FS transcript.** `fs_on_host` keeps the sponge
    host-resident as an eager primitive, so it cannot be traced.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")

    state = RunningClaim(claim, fnp.zeros((proof.shape[0],), claim.dtype), fnp.int32(0))
    oks = []
    for i in range(proof.shape[0]):
        state, transcript, ok = verifier(state, transcript, proof[i])
        oks.append(ok)
    return state.point, state.value, transcript, fnp.all(fnp.stack(oks))
