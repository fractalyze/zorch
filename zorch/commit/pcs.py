# zorch/commit/pcs.py
# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The multilinear-evaluation PCS interface.

A `Pcs` commits to a multilinear polynomial given by its evaluations over the
boolean hypercube (an MLE `[2^v, w]`; the `w` columns share the domain) and
later opens it at an evaluation point. `commit` is the only method P2
implements; `open` / `verify` complete the evaluation argument and land in P3.

The `(commitment, prover_data)` split is the load-bearing contract:

- `commitment` is the succinct public value (a Merkle root). It is what enters
  the Fiat-Shamir transcript and what the verifier receives.
- `prover_data` is the retained witness `open` consumes (the full Merkle tree /
  codeword plus metadata). It is never sent to the verifier.

This split is what lets `commit` be transcript-observable and `open` be a pure
function of retained prover state. See `docs/pcs.md` for the full design rules.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Pcs(Protocol):
    def commit(self, mle: Array) -> tuple[Array, Any]:
        """Commit an MLE `[2^v, w]`; return `(commitment, prover_data)`."""
        ...

    # --- P3: evaluation argument (provisional signatures) ----------------
    def open(
        self, prover_data: Any, point: Array, transcript: Any
    ) -> tuple[Array, Any]:
        """Open the committed MLE at `point`; return `(value, proof)`. (P3)"""
        ...

    def verify(
        self,
        commitment: Array,
        point: Array,
        value: Array,
        proof: Any,
        transcript: Any,
    ) -> bool:
        """Check `proof` that the committed MLE evaluates to `value`. (P3/P4)"""
        ...
