# zorch/commit/jagged/commit.py
# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged commit: pack blocks into the stacked MLE -> `pcs.commit` -> bind the
row/column structure into the commitment.

`from_blocks` (host) packs the blocks into the static `M_max=2^tier` buffer; the
device commit (`pcs.commit` -> bind) is one `@jit` zone keyed on the MLE +
structure shapes. It compiles once per **(area tier, block count)**: the MLE is
`[S, K]` (fixed by the tier) and the structure vector is `(1 + 2*num_blocks,)`.
zorch's `from_blocks` keeps empty blocks (does not skip them), so the block list
is structurally fixed for a given consumer layout — block *heights* varying
within a tier reuse the same compiled artifact, the AOT property that matters.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zorch.commit.jagged.dense import JaggedLayout, from_blocks
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.protocol import PcsProver


def _structure_vec(layout: JaggedLayout) -> Array:
    """[num_blocks, heights..., widths...] as a base-field row, for binding.

    Counts are bounded by M_max = 2^log_m < p for all supported field primes,
    so they round-trip through the field dtype without modular reduction.
    """
    vals = [len(layout.heights), *layout.heights, *layout.widths]
    return jnp.asarray(vals, dtype=layout.dtype)


class JaggedPcs:
    """Commits variable-height blocks via an injected multilinear `PcsProver`,
    binding the jagged structure (row/column counts) into the returned commitment.

    `sponge` / `compressor` perform the structure bind (the jagged layer owns it,
    not the PCS). The jit'd device zone is built once in `__init__`, capturing
    `prover` / `sponge` / `compressor` constants, and exposed as `self._commit`. It
    is keyed on the MLE + structure shapes, so it compiles once per **(area tier,
    block count)** and reuses the artifact as block heights vary within a tier;
    making it strictly block-count-invariant would need a fixed-length structure
    vector, deferred as unnecessary for now.
    """

    _commit: Any  # jax.jit-wrapped device-commit closure, built in __init__

    def __init__(
        self, prover: PcsProver, sponge: Sponge, compressor: Compression
    ) -> None:
        if sponge.out != compressor.chunk:
            raise ValueError(
                f"sponge digest ({sponge.out}) must equal compressor chunk "
                f"({compressor.chunk}) for the structure bind"
            )
        if compressor.arity != 2:
            raise ValueError(
                f"compressor.arity must be 2 for the (root, structure_hash) "
                f"bind, got {compressor.arity}"
            )
        self.prover = prover
        self.sponge = sponge
        self.compressor = compressor

        @jax.jit
        def _commit(mle: Array, structure: Array) -> tuple[Array, Any]:
            # The seam commits a Sequence of column MLEs; BaseFold binds them
            # jointly under one root (a matrix commitment), which the structure
            # bind below hashes against.
            columns = [mle[:, j] for j in range(mle.shape[1])]
            root, prover_data = prover.commit(columns)
            structure_hash = sponge.hash(structure)
            bound = compressor.compress(jnp.stack([root, structure_hash]))
            return bound, prover_data

        self._commit = _commit

    def commit(
        self, blocks: list[Array], *, log_stacking_height: int
    ) -> tuple[Array, Any]:
        """Pack + commit; return `(commitment, prover_data)`.

        `from_blocks` (host) derives the tier and the stacked MLE `[S, K]`; the
        `@jit` zone commits it via `pcs` and binds the structure hash into the
        root.
        """
        packed, layout = from_blocks(blocks, log_stacking_height=log_stacking_height)
        mle = packed.reshape(layout.K, layout.S).T  # [S, K]
        structure = _structure_vec(layout)
        return self._commit(mle, structure)
