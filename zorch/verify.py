# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generic verifier driver — the dual of `prove`.

The per-round check and claim-reduction live on the verifier round; this driver
only scans it over the proof, threads the transcript, collects the challenge
point, and ANDs the consistency flags. It stops at the reduced point-claim — the
final `final_claim == oracle(point)` check needs a PCS opening and is the
consumer's, keeping this block proving-scheme- and PCS-agnostic.

The replay is one `lax.scan` over the proof rows (carry `(claim, transcript)`),
not a Python loop: the whole verification compiles to a single traced region that
is flat in the round count (issue #58), so it stays one fused unit rather than an
unrolled body that crosses the ZKX PTX cliff.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax

from zorch.round import Round
from zorch.transcript import Transcript


def verify(
    verifier: Round, claim: Array, proof: Array, transcript: Transcript
) -> tuple[Array, Array, Transcript, Array]:
    """Replay `proof` against `claim` → `(point, final_claim, transcript, ok)`.

    `ok` ANDs every round's check; one false anywhere rejects the proof.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")

    def step(
        carry: tuple[Array, Transcript], msg: Array
    ) -> tuple[tuple[Array, Transcript], tuple[Array, Array]]:
        claim, transcript = carry
        claim, transcript, r, ok = verifier(claim, msg, transcript)
        return (claim, transcript), (r, ok)

    (claim, transcript), (point, oks) = lax.scan(step, (claim, transcript), proof)
    return point, claim, transcript, jnp.all(oks)
