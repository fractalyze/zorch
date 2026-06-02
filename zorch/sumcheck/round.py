# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sumcheck as a `Round`: product sumcheck, one round per call.

State is `list[Array]` — raw MLE evals, one per factor; the round sums their
product over the hypercube. The round body is element-wise field ops + the one
inherent Sigma (no reduce/gather beyond it), so it stays wrappable by the
Phase-3 single-kernel marker without restructuring (no fusion decorator now).

A non-product summand (e.g. zerocheck eq*(A*B-C), LogUp) is a generalization to
add when its first consumer lands -- not carried speculatively here.
"""
from __future__ import annotations

import operator
from functools import reduce

import jax.numpy as jnp
from jax import Array

from zorch.round import Round


class SumcheckRound(Round):
    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

    def _split(self, state):
        out = []
        for evals in state:
            half = evals.shape[-1] // 2
            out.append((evals[..., :half], evals[..., half:]))
        return out

    def round_poly(self, state) -> Array:
        """Round polynomial over the domain [0, 1, ..., degree], shape
        (degree+1,): s[u] = sum_x' prod_k (P0_k + u*(P1_k - P0_k)).

        The whole u-domain is evaluated at once — one batched reduction, so it
        lowers toward a single reduction kernel rather than degree+1 separate
        ones. The u-domain is built with jnp.stack (not jnp.arange, whose iota
        is unsupported for extension dtypes)."""
        halves = self._split(state)
        dtype = state[0].dtype
        us = jnp.stack([jnp.array(u, dtype) for u in range(self.degree + 1)])
        factors = (p0 + us[:, None] * (p1 - p0) for (p0, p1) in halves)
        return jnp.sum(reduce(operator.mul, factors), axis=-1)

    def fold(self, state, r) -> list:
        """Fold each MLE at challenge `r`: P0 + r*(P1 - P0). Halves width."""
        return [p0 + r * (p1 - p0) for (p0, p1) in self._split(state)]

    def __call__(self, state, transcript):
        msg = self.round_poly(state)
        transcript = self.commit(transcript, msg)
        transcript, r = self.challenge(transcript, 1)
        state = self.fold(state, r[0])
        return state, transcript, msg
