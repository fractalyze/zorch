# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Staggered partial-Lagrange batching for the BaseFold batch open.

A batch open reduces several separately committed matrices (e.g. a preprocessed
and a main region) to one FRI: their columns are combined into a single codeword
by a random linear combination, the fold chain runs once on the combination, and
each query opens every matrix at the shared positions. The weights are the
partial-Lagrange basis `eq(., r)` over `num_batch_vars = log2_ceil(total_width)`
challenges, allocated *staggered* across the rounds — round 0's columns take the
first `w_0` weights, round 1's the next `w_1`, and so on. The single-round,
single-width case collapses to the `[1]` weight, so the degenerate batch is the
plain open.

The partial-Lagrange weights are `eq(., r)` (the same basis `eval_mle` folds on),
not powers of one challenge: a powers RLC would round-trip internally but cannot
match a consumer that derives its batch weights this way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import frx.numpy as jnp
from frx import Array

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize


def partial_lagrange(point: Array) -> Array:
    """The partial-Lagrange basis `eq(., point)` over `2^m` hypercube points for
    an `m`-dimensional `point` — the batch weights' source. `m == 0` -> `[1]`."""
    return expand_eq_to_hypercube(point, jnp.ones((), point.dtype))


def sample_staggered_coeffs(
    transcript: Transcript, total_width: int, dtype: Any, *, lsb_first: bool = False
) -> tuple[Transcript, Array]:
    """Batch weights for `total_width` columns: `log2_ceil(total_width)` squeezed
    challenges expanded to the partial-Lagrange basis (length `2^nbv >= total_width`).
    `total_width == 1` -> `[1]`, no squeeze. Called by both `open` and `verify` so
    the two sides derive identical weights.

    `lsb_first` picks the table's index orientation. False keeps
    `partial_lagrange`'s native MSB-first indexing (challenge `j` <-> index bit
    `nbv-1-j`); True reverses the challenge vector before expansion so challenge
    `j` <-> index bit `j`. A wire-format convention knob: for non-power-of-two
    widths the leading `total_width` weights are a *different set* per
    orientation, so a consumer whose eq tables index LSB-first cannot be matched
    by reordering after the fact. Transcript consumption is identical either way.
    """
    nbv = log2_ceil_usize(total_width)
    if nbv == 0:
        return transcript, jnp.ones(1, dtype)
    transcript, s = transcript.sample(nbv)
    s = s.astype(dtype)
    return transcript, partial_lagrange(s[::-1] if lsb_first else s)


def batch_staggered(columns: Sequence[Array], coeffs: Array) -> Array:
    """Staggered RLC of per-round column matrices into one array, summed along a
    shared leading axis. `columns[r]` is `[..., w_r]`; round `r` consumes
    `coeffs[offset : offset + w_r]`, `offset` advancing by each round's width.
    Returns the `[...]` weighted column sum across every round."""
    if not columns:
        raise ValueError("batch_staggered needs at least one round of columns")
    acc = jnp.zeros(columns[0].shape[:-1], dtype=columns[0].dtype)
    offset = 0
    for col in columns:
        w = col.shape[-1]
        acc = acc + (col * coeffs[offset : offset + w]).sum(axis=-1)
        offset += w
    return acc
