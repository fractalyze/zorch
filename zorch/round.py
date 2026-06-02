# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The composable IOP `Round` — zorch's nn.Module-style building block.

Subclass and implement `__call__`, using the two Fiat-Shamir primitives the base
provides. A prover round maps `(state, transcript) -> (state, transcript, msg)`;
its verifier dual maps `(claim, msg, transcript) -> (claim, transcript, r, ok)`.
The transcript is an immutable carry threaded functionally (never hidden mutable
state).
"""
from __future__ import annotations

from jax import Array

from zorch.transcript import Transcript


class Round:
    def observe(self, transcript: Transcript, values: Array) -> Transcript:
        """Prover sends a message for the transcript to observe."""
        return transcript.observe(values)

    def challenge(self, transcript: Transcript, n: int = 1):
        """Sample `n` challenges from the transcript."""
        return transcript.sample(n)

    def __call__(self, *args):
        """Run one round, threading the transcript — see the module docstring for
        the prover and verifier signatures. Implemented by subclasses."""
        raise NotImplementedError
