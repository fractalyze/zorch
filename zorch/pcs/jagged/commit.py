# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are RS-encoded
(``BitReversedReedSolomon``) into a ``[S*blowup, K]`` bit-reversed codeword,
Merkle-committed via the SMCS, then bound to the region's row/column structure.
The commit half of the jagged PCS — it produces the ``StackedRound`` the stacked
open (``zorch.pcs.jagged.open``) consumes. ``jit=True`` fuses the commit tail to
one zone (required at rsp scale; see ``_commit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.region import JaggedRegion


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "dense",
        "mle",
        "codeword",
        "digest_layers",
        "row_counts",
        "column_counts",
        "smcs_commitment",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class TraceCommitData:
    """Prover-side witness the opening stage retains: ``mle`` (``[S,K]`` message)
    and ``codeword`` (``[S*blowup,K]`` bit-reversed leaves), ``None`` under
    ``drop_codeword`` (the open re-encodes from ``mle``). The row/column counts
    are kept because the structure hash bound them and the verifier rebind needs
    the exact device values."""

    dense: Array
    mle: Array
    codeword: Array | None
    digest_layers: list[Array]
    row_counts: Array
    column_counts: Array
    smcs_commitment: Array  # shape-bound SMCS root, before structure binding


def _commit(
    message: Array,
    row_counts: Array,
    column_counts: Array,
    *,
    smcs: SingleMatrixCommitmentScheme,
    log_blowup: int,
    drop_codeword: bool = False,
) -> tuple[Array, Array, Array | None, list[Array], Array]:
    """The device-side commit, shared by the eager and @jit paths. Folding the
    structure-bind poseidon2 into the Merkle-commit zone avoids its per-eager-call
    composite recompile (seconds at rsp scale). ``code`` is rebuilt per call, not
    a static arg: it is identity-hashed, so a fresh instance would recompile the
    zone; construction is attribute-only (free under trace)."""
    code = BitReversedReedSolomon(
        message_len=message.shape[-1], blowup=1 << log_blowup, dtype=message.dtype
    )
    # Commit COLUMN-major: a leaf is a column of the native [K, N] encode, so the
    # SMCS leaf-hash skips the codeword transpose (sp1-zorch#140); byte-identical
    # root (leaf r = column r of [K,N] = row r of the [N,K] leaf-major view).
    codeword = code.encode(message)
    commitment, digest_layers = smcs.commit(codeword, column_major=True)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    # Retain the codeword in the open's [N, K] leaf-major layout (transpose, kept
    # only when wanted). ``drop_codeword`` (SP1 drop_ldes) omits the ~6 GB blow-up
    # from the outputs; the open re-encodes it from ``mle`` (sp1-zorch#55, #124).
    out_codeword = None if drop_codeword else codeword.T
    return bound, message.T, out_codeword, digest_layers, commitment


# ``smcs`` is a static arg keyed by object identity (the scheme defines no
# __eq__/__hash__): every call site must reuse one instance per process, or
# each fresh instance silently recompiles the full poseidon2/Merkle pipeline.
_commit_jit = jax.jit(_commit, static_argnames=("smcs", "log_blowup", "drop_codeword"))


def commit_region(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    *,
    log_blowup: int,
    jit: bool = False,
    drop_codeword: bool = False,
) -> tuple[Array, TraceCommitData]:
    """Commit a packed region; returns ``(bound_commitment, prover_data)``.

    ``jit`` runs the commit tail as one fused graph — required at rsp scale
    on a 32 GB device (see the module docstring). Byte-identical either way.

    ``drop_codeword`` (SP1's drop_ldes) returns ``TraceCommitData.codeword =
    None`` and never materializes the ~6 GB blow-up as an output, so it does not
    stay device-resident through the chain; the open re-encodes it from ``mle``.
    """
    S = 1 << region.log_stacking_height
    dense = region.dense
    if dense.shape[0] % S != 0:
        raise ValueError(
            f"dense size {dense.shape[0]} must be a multiple of the stacking "
            f"height {S} (from_chips pads to it)"
        )
    K = dense.shape[0] // S

    # Row k of [K, S] is stacked column k of the dense MLE.
    message = dense.reshape(K, S)
    row_counts = jnp.array(region.row_counts, dtype=dense.dtype)
    column_counts = jnp.array(region.column_counts, dtype=dense.dtype)
    tail = _commit_jit if jit else _commit
    bound, mle, codeword_t, digest_layers, commitment = tail(
        message,
        row_counts,
        column_counts,
        smcs=smcs,
        log_blowup=log_blowup,
        drop_codeword=drop_codeword,
    )
    return bound, TraceCommitData(
        dense=dense,
        mle=mle,
        codeword=codeword_t,
        digest_layers=digest_layers,
        row_counts=row_counts,
        column_counts=column_counts,
        smcs_commitment=commitment,
    )
