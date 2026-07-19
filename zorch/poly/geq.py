# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Virtual ``x >= threshold`` indicator over the boolean hypercube."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["threshold", "geq_coefficient", "eq_coefficient"],
    meta_fields=[],
)
@dataclass(frozen=True)
class VirtualGeq:
    """Compact ``geq_{>=threshold}``: entries below ``threshold`` read 0, the
    entry AT ``threshold`` reads ``eq_coefficient + geq_coefficient``, entries
    above read ``geq_coefficient``.

    Folding one variable keeps this three-zone closed form (the straddling
    pair is what forces the separate at-threshold coefficient), so the
    indicator threads through a per-variable fold loop without ever
    materializing a ``2**num_vars`` vector.

    ``threshold`` is a traced leaf, not static config: a jagged sumcheck round
    engine carries the indicator as a ``lax.scan`` carry whose threshold halves
    each round, so the zone branches are value-generic ``fnp.where`` (a Python
    ``if`` on the threshold breaks under the scan). A Python ``int`` at
    construction is coerced, so the static call sites stay byte-identical."""

    threshold: Array
    geq_coefficient: Array
    eq_coefficient: Array

    def fix_last_variable(self, alpha: Array) -> VirtualGeq:
        """Bind the LSB variable at ``alpha`` — the partial-eval bind
        ``(1-alpha)*e0 + alpha*e1`` (NOT ``mle_fold``'s additive combine).
        Pairs strictly past the threshold bind to ``geq``; the straddling
        pair becomes the new ``eq_coefficient``, by which side of the pair
        the threshold sits on."""
        one = fnp.ones((), alpha.dtype)
        threshold = fnp.asarray(self.threshold, fnp.int32)
        # Threshold even — pair (threshold, threshold+1) = (eq+geq, geq):
        # eq+geq + alpha*(geq - (eq+geq)) = geq + (1-alpha)*eq.
        even_eq = (one - alpha) * self.eq_coefficient
        # Threshold odd — pair (threshold-1, threshold) = (0, eq+geq) binds to
        # alpha*(eq+geq); the bound index sits AT the new threshold whose
        # "above" zone reads geq, so the eq part is alpha*(eq+geq) - geq.
        odd_eq = (
            alpha * (self.eq_coefficient + self.geq_coefficient) - self.geq_coefficient
        )
        new_eq = fnp.where(threshold % 2 == 0, even_eq, odd_eq)
        return VirtualGeq(threshold >> 1, self.geq_coefficient, new_eq)

    def eval_at(self, index: Array | int) -> Array:
        threshold = fnp.asarray(self.threshold, fnp.int32)
        index = fnp.asarray(index, fnp.int32)
        at_or_above = fnp.where(
            index == threshold,
            self.eq_coefficient + self.geq_coefficient,
            self.geq_coefficient,
        )
        return fnp.where(
            index < threshold, fnp.zeros_like(self.geq_coefficient), at_or_above
        )
