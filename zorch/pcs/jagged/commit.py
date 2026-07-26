# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are RS-encoded
(``BitReversedReedSolomon``) into a ``[S*blowup, K]`` bit-reversed codeword,
Merkle-committed via the SMCS, then bound to the region's row/column structure.
The commit half of the jagged PCS — it produces the ``StackedRound`` the stacked
open (``zorch.pcs.jagged.open``) consumes.

``jit=True`` runs the commit as two ``@jit`` zones cut on the K (stacked column
count) compile key — the ``stacked_basefold_open`` zoning recipe. Only the
encode + leaf-hash prologue's shapes carry K; the Merkle fold's O(depth)
compile — the dominant one — keys on the leaf count ``S*blowup`` alone, so it
compiles once per process and is shared by every shard. Byte-identical to eager
either way (the zone cut sits on the leaf-digest layer both paths compute).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.region import JaggedRegion
from zorch.utils.bits import log2_strict_usize


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "dense",
        "mle",
        "digest_layers",
        "row_counts",
        "column_counts",
        "smcs_commitment",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class TraceCommitData:
    """Prover-side witness the opening stage retains: the ``[S,K]`` message
    ``mle`` and the digest tree. The codeword is not kept — the open re-encodes
    it from ``mle``. The row/column counts are kept because the structure hash
    bound them and the verifier rebind needs the exact device values."""

    dense: Array
    mle: Array
    digest_layers: list[Array]
    row_counts: Array
    column_counts: Array
    smcs_commitment: Array  # shape-bound SMCS root, before structure binding


# K-shaped zone: encode + leaf hash, the only shapes carrying K. Cheap to
# recompile per K (one NTT + one fused sponge region), and the ~6 GB codeword
# lives and dies here rather than crossing into the fold zone. ``code`` is
# rebuilt per call, not passed in: it hashes by value, so a fresh instance
# never forces a recompile. Commits COLUMN-major so a leaf is a column of the
# native [K, N] encode (skips the codeword transpose).
def _prologue(
    smcs: SingleMatrixCommitmentScheme, message: Array, *, log_blowup: int
) -> tuple[Array, Array]:
    code = BitReversedReedSolomon(
        message_len=message.shape[-1], blowup=1 << log_blowup, dtype=message.dtype
    )
    codeword = code.encode(message)
    return message.T, smcs.hash_leaves(codeword, column_major=True)


# K-independent zone: the O(depth) Merkle fold plus the root/structure binds —
# the commit's dominant compile. No shape here carries K (the leaf-digest layer
# is [S*blowup, digest_elems] and K enters the separator preimage as the traced
# ``shape_params`` value), so it compiles once and is shared by every shard of
# one (S, blowup, chip-count) configuration.
def _fold_bind(
    smcs: SingleMatrixCommitmentScheme,
    leaf_digests: Array,
    shape_params: Array,
    row_counts: Array,
    column_counts: Array,
) -> tuple[Array, list[Array], Array]:
    raw_root, digest_layers = smcs.fold_leaf_digests(leaf_digests)
    commitment = smcs.bind_root(raw_root, shape_params)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    return bound, digest_layers, commitment


# ``smcs`` keys these zones by object identity (the scheme defines no
# __eq__/__hash__), so every call site must reuse one instance per process or
# each fresh instance silently recompiles the whole poseidon2/Merkle pipeline.
_prologue_jit = frx.jit(_prologue, static_argnames=("smcs", "log_blowup"))
_fold_bind_jit = frx.jit(_fold_bind, static_argnames=("smcs",))


def commit_region(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    *,
    log_blowup: int,
    jit: bool = False,
) -> tuple[Array, TraceCommitData]:
    """Commit a packed region; returns ``(bound_commitment, prover_data)``.

    ``jit`` fuses each zone (the module docstring) — required at rsp scale on a
    32 GB device; eager runs the same two bodies un-fused. Byte-identical either
    way. The ~6 GB blow-up codeword never leaves this function: the open
    re-encodes it from ``mle``, so it never stays device-resident."""
    message = region.block
    row_counts = fnp.array(region.row_counts, dtype=message.dtype)
    column_counts = fnp.array(region.column_counts, dtype=message.dtype)
    K, S = message.shape
    # The [log_height, width] separator preimage as a value, so the fold zone
    # reads K as data rather than a compile key.
    shape_params = fnp.array(
        [log2_strict_usize(S << log_blowup), K], dtype=message.dtype
    )

    mle, leaf_digests = (_prologue_jit if jit else _prologue)(
        smcs, message, log_blowup=log_blowup
    )
    bound, digest_layers, commitment = (_fold_bind_jit if jit else _fold_bind)(
        smcs, leaf_digests, shape_params, row_counts, column_counts
    )
    return bound, TraceCommitData(
        dense=region.dense,
        mle=mle,
        digest_layers=digest_layers,
        row_counts=row_counts,
        column_counts=column_counts,
        smcs_commitment=commitment,
    )
