# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generic verifier driver — the dual of `prove`.

The per-round check and claim-reduction live on the verifier round; this driver
only loops it over the proof, threads the transcript, collects the challenge
point, and ANDs the consistency flags. It stops at the reduced point-claim — the
final `final_claim == oracle(point)` check needs a PCS opening and is the
consumer's, keeping this block proving-scheme- and PCS-agnostic.
"""
from __future__ import annotations

import jax.numpy as jnp

from zorch.round import Round


def verify(verifier: Round, claim, proof, transcript):
    """Replay `proof` against `claim` → `(point, final_claim, transcript, ok)`.

    `ok` ANDs every round's check; one false anywhere rejects the proof.
    """
    if proof.ndim != 2 or proof.shape[0] == 0:
        raise ValueError("proof must be a non-empty 2-D array (one row per round)")
    ok = jnp.array(True)
    point = []
    for msg in proof:
        claim, transcript, r, ok_i = verifier(claim, msg, transcript)
        ok = ok & ok_i
        point.append(r)
    return jnp.stack(point), claim, transcript, ok
