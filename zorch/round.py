# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The composable IOP `Round` — zorch's nn.Module-style building block.

Subclass and implement `__call__`, threading the transcript and calling its
`observe` / `sample` directly. A prover round maps
`(state, transcript) -> (state, transcript, msg)`; its verifier dual maps
`(claim, msg, transcript) -> (claim, transcript, r, ok)`. The transcript is an
immutable carry threaded functionally (never hidden mutable state).
"""
from __future__ import annotations


class Round:
    def __call__(self, *args):
        """Run one round, threading the transcript — see the module docstring for
        the prover and verifier signatures. Implemented by subclasses."""
        raise NotImplementedError
