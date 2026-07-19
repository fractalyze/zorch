# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Witness-opening glue: the sumcheck-claim → PCS-opening combinator.

The piece `zorch.verify` leaves to the consumer ("the verifier reduces; the PCS
closes"). It closes the lincheck's `inner_final == eval_ABC · z̃(r_y)`:
`WitnessOpenProver` opens `W` (the low half of `z`) at `r_y[1:]`;
`WitnessOpenVerifier` verifies that, reconstructs `z̃(r_y)` from the opened
`eval_W` and the verifier-evaluated public half, and checks the identity.

The glue touches only the `zorch.pcs.protocol` seam, so the PCS is injected —
`DensePcs` is a transparent reference instance for tests; a real `basefold` /
`whir` / `kzg` pair drops in unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.pcs.protocol import PcsProver, PcsVerifier
from zorch.poly.multilinear import eval_mle
from zorch.round import Stage
from zorch.spartan.carry import SpartanCarry, _require
from zorch.spartan.r1cs import R1CS, eval_public_half, recombine_z_eval
from zorch.transcript import Transcript


class WitnessOpenProver(Stage):
    """Open the committed witness at `r_y[1:]` via the injected `PcsProver`."""

    def __init__(self, pcs: PcsProver[Any, Any, Any], prover_data: Any) -> None:
        self.pcs = pcs
        self.prover_data = prover_data

    def __call__(
        self, carry: SpartanCarry, transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, tuple[Array, object]]:
        r_y = _require(carry.r_y, "r_y", "inner")
        values, proof, transcript = self.pcs.open(
            self.prover_data, [r_y[1:]], transcript
        )
        return carry, transcript, (values, proof)


class WitnessOpenVerifier(Stage):
    """Verify the witness opening and close the lincheck identity."""

    def __init__(
        self,
        pcs: PcsVerifier[Any, Any],
        commitment: Array,
        instance: R1CS,
        io: Array,
    ) -> None:
        self.pcs = pcs
        self.commitment = commitment
        self.instance = instance
        self.io = io

    def __call__(
        self, carry: SpartanCarry, msg: tuple[Array, object], transcript: Transcript
    ) -> tuple[SpartanCarry, Transcript, Array]:
        values, proof = msg
        r_x = _require(carry.r_x, "r_x", "outer")
        r = _require(carry.r_batch, "r_batch", "RLC")
        r_y = _require(carry.r_y, "r_y", "inner")
        inner_final = _require(carry.inner_final, "inner_final", "inner")

        ok_open, transcript = self.pcs.verify(
            self.commitment, [r_y[1:]], values, proof, transcript
        )
        eval_w = values[0]
        eval_pub = eval_public_half(self.io, r_y[1:], self.instance.num_vars_padded)
        z_eval = recombine_z_eval(eval_w, eval_pub, r_y[0])
        eval_abc = self.instance.eval_combined_matrix(r_x, r_y, r)
        ok_final = inner_final == eval_abc * z_eval
        return carry, transcript, ok_open & ok_final


class DensePcs:
    """A transparent multilinear "PCS" for tests: the commitment IS the evals.

    Satisfies `PcsProver` and `PcsVerifier` — `open` evaluates the MLE, `verify`
    recomputes it from the commitment. Neither hiding nor succinct; it exists only
    to exercise the glue against the seam without a FRI backend.
    """

    def commit(self, polys: Sequence[Array]) -> tuple[Array, tuple[Array, ...]]:
        # One committed poly in Spartan (the witness); the commitment is its evals.
        return polys[0], tuple(polys)

    def open(
        self,
        prover_data: tuple[Array, ...],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, None, Transcript]:
        poly = prover_data[0]
        values = fnp.stack([eval_mle(poly, pt) for pt in points])
        return values, None, transcript

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: None,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        del proof
        recomputed = fnp.stack([eval_mle(commitment, pt) for pt in points])
        return fnp.all(recomputed == values), transcript
