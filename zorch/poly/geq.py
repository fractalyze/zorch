# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Virtual ``x >= threshold`` indicator over the boolean hypercube."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["geq_coefficient", "eq_coefficient"],
    meta_fields=["threshold"],
)
@dataclass(frozen=True)
class VirtualGeq:
    """Compact ``geq_{>=threshold}``: entries below ``threshold`` read 0, the
    entry AT ``threshold`` reads ``eq_coefficient + geq_coefficient``, entries
    above read ``geq_coefficient``.

    Folding one variable keeps this three-zone closed form (the straddling
    pair is what forces the separate at-threshold coefficient), so the
    indicator threads through a per-variable fold loop without ever
    materializing a ``2**num_vars`` vector."""

    threshold: int
    geq_coefficient: Array
    eq_coefficient: Array

    def fix_last_variable(self, alpha: Array) -> VirtualGeq:
        """Bind the LSB variable at ``alpha`` — the partial-eval bind
        ``(1-alpha)*e0 + alpha*e1`` (NOT ``mle_fold``'s additive combine).
        Pairs strictly past the threshold bind to ``geq``; the straddling
        pair becomes the new ``eq_coefficient``, by which side of the pair
        the threshold sits on."""
        one = jnp.ones((), alpha.dtype)
        if self.threshold & 1 == 0:
            # Pair (threshold, threshold+1) = (eq+geq, geq):
            # eq+geq + alpha*(geq - (eq+geq)) = geq + (1-alpha)*eq.
            new_eq = (one - alpha) * self.eq_coefficient
        else:
            # Pair (threshold-1, threshold) = (0, eq+geq) binds to
            # alpha*(eq+geq); the bound index sits AT the new threshold whose
            # "above" zone reads geq, so the eq part is alpha*(eq+geq) - geq.
            new_eq = (
                alpha * (self.eq_coefficient + self.geq_coefficient)
                - self.geq_coefficient
            )
        return VirtualGeq(self.threshold >> 1, self.geq_coefficient, new_eq)

    def eval_at(self, index: int) -> Array:
        if index < self.threshold:
            return jnp.zeros((), self.geq_coefficient.dtype)
        if index == self.threshold:
            return self.eq_coefficient + self.geq_coefficient
        return self.geq_coefficient
