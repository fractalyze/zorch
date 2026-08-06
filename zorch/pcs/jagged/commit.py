# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are RS-encoded
(``BitReversedReedSolomon``) into a ``[S*blowup, K]`` bit-reversed codeword,
Merkle-committed via the SMCS, then bound to the region's row/column structure.
The commit half of the jagged PCS — it produces the ``StackedRound`` the stacked
open (``zorch.pcs.jagged.open``) consumes.

``jit=True`` runs the commit as three ``@jit`` zones — the
``stacked_basefold_open`` zoning recipe. Only the encode + leaf-hash prologue's
shapes carry K (the stacked column count); the Merkle fold's O(depth) compile —
the dominant one — keys on the leaf count ``S*blowup`` plus the
identity-hashed ``smcs`` static, so it compiles once per leaf count for each
long-lived ``SingleMatrixCommitmentScheme`` (the shard prover holds one for
its lifetime; a fresh instance recompiles) and is shared by every shard of
that height; the root/structure bind tail —
the only counts-shaped work, a two-permute graph — recompiles per chip count
without ever touching the fold. Byte-identical to eager either way (the zone
cuts sit on the leaf-digest and raw-root layers both paths compute).
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


# Leaf-count zone: the O(depth) Merkle fold — the commit's dominant compile.
# The only shape here is the [S*blowup, digest_elems] leaf-digest layer, so it
# keys on the leaf count alone and compiles once per (S, blowup) — shared by
# every shard regardless of K or chip count. Keeping the counts-shaped bind
# tail out is what makes that true: with it inside, every distinct chip count
# re-paid this whole O(depth) graph for a two-permute tail.
def _fold(
    smcs: SingleMatrixCommitmentScheme, leaf_digests: Array
) -> tuple[Array, list[Array]]:
    return smcs.fold_leaf_digests(leaf_digests)


# Counts-shaped zone: the root/structure binds. Recompiles per chip count
# (the row/column-count length), but its graph is two sponge hashes + two
# compressions over the 8-element raw root, so that compile is trivial. K
# enters the separator preimage as the traced ``shape_params`` value, never as
# a compile key.
def _bind(
    smcs: SingleMatrixCommitmentScheme,
    raw_root: Array,
    shape_params: Array,
    row_counts: Array,
    column_counts: Array,
) -> tuple[Array, Array]:
    commitment = smcs.bind_root(raw_root, shape_params)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    return bound, commitment


# ``smcs`` keys these zones by object identity (the scheme defines no
# __eq__/__hash__), so every call site must reuse one instance per process or
# each fresh instance silently recompiles the whole poseidon2/Merkle pipeline.
_prologue_jit = frx.jit(_prologue, static_argnames=("smcs", "log_blowup"))
_fold_jit = frx.jit(_fold, static_argnames=("smcs",))
_bind_jit = frx.jit(_bind, static_argnames=("smcs",))


def commit_region(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    *,
    log_blowup: int,
    jit: bool = False,
) -> tuple[Array, TraceCommitData]:
    """Commit a packed region; returns ``(bound_commitment, prover_data)``.

    ``jit`` fuses each zone (the module docstring) — required at rsp scale on a
    32 GB device; eager runs the same three bodies un-fused. Byte-identical either
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
    raw_root, digest_layers = (_fold_jit if jit else _fold)(smcs, leaf_digests)
    bound, commitment = (_bind_jit if jit else _bind)(
        smcs, raw_root, shape_params, row_counts, column_counts
    )
    return bound, TraceCommitData(
        dense=region.dense,
        mle=mle,
        digest_layers=digest_layers,
        row_counts=row_counts,
        column_counts=column_counts,
        smcs_commitment=commitment,
    )
