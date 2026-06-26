# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are
RS-encoded (zorch ``BitReversedReedSolomon``) into a ``[S * blowup, K]``
codeword in SP1's bit-reversed row order, Merkle-committed via the SMCS,
then bound to the region's row/column structure. Mirrors sp1-hypercube's
jagged commit (the prover half of the basefold commitment SP1 core uses).
This is the commit half of the jagged PCS — it produces the ``StackedRound``
the stacked open (``zorch.pcs.jagged.open``) consumes.

``jit=True`` runs the whole commit -- RS-encode -> Merkle -> structure bind
-- as one @jit zone. Two reasons: the eager form keeps every codeword-scale
intermediate (~6 GB at rsp scale) live at once and OOMs a 32 GB device,
while one fused graph lets XLA release each buffer after its last use; and
the structure-bind's poseidon2 hash recompiles its composite on every eager
call (seconds at rsp scale for trivial arithmetic), which folding it into
the zone removes. No input is donated -- the dense buffer outlives the
commit (the jagged-eval stage reads it again), so there is nothing safe to
donate. Output is byte-identical either way.
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


def committed_codeword(code: BitReversedReedSolomon, message: Array) -> Array:
    """The committed codeword in the open's leaf-major ``[S*blowup, K]`` layout:
    the bit-reversed RS encode of the ``[K, S]`` message, transposed.

    The open's ``drop_ldes`` re-encode (``zorch.pcs.jagged.open``) recovers the
    codeword through this transform so its Merkle query paths authenticate
    against the commitment. The commit itself reads the pre-transpose
    ``[K, S*blowup]`` encode column-major (``_commit``) — same leaf content, no
    transpose — so this leaf-major form is the open side only.
    """
    return code.encode(message).T


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
    """Prover-side retained state for the opening stage.

    The row/column count arrays live here because the structure hash bound
    them; the verifier-side rebind needs the exact device values. ``mle`` and
    ``codeword`` are the stacked open's per-region witness — the ``[S, K]``
    message matrix and the committed ``[S*blowup, K]`` bit-reversed leaves —
    so the opening stage reproves at the eval point without recommitting.
    ``codeword`` is ``None`` when committed with ``drop_codeword=True`` (SP1's
    drop_ldes): the open re-encodes it from ``mle`` rather than holding the
    ~6 GB blow-up device-resident through the chain.
    """

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
    """The whole device-side commit, shared by the eager and @jit paths.

    Everything that touches poseidon2 lives in one program: the structure bind
    is folded in next to the Merkle commit, not run eagerly afterward. Eager
    poseidon2 recompiles its composite per call, so the ~10-permutation
    structure hash costs seconds outside @jit even though its arithmetic is
    trivial — under @jit it compiles once with the Merkle tree and executes in
    microseconds. The ``BitReversedReedSolomon`` is rebuilt per call rather than
    passed in: it is identity-hashed (no __eq__/__hash__), so a per-call
    instance as a static arg would recompile the zone every call, and
    construction without a coset shift is attribute-only — free under trace.
    """
    # SP1's codeword layout is bit-reversed (FRI fold pairs adjacent) — the
    # same code object the opening stage folds with.
    code = BitReversedReedSolomon(
        message_len=message.shape[-1], blowup=1 << log_blowup, dtype=message.dtype
    )
    # Commit the codeword COLUMN-major: a leaf is a column of the native
    # [K, S*blowup] encode, so the SMCS leaf-hash reads it directly (SP1's
    # absorbRow) and the fused commit skips the codeword transpose that
    # dominated it (fractalyze/sp1-zorch#140). Byte-identical root — leaf r is
    # column r of [K, N] == row r of the [N, K] leaf-major view.
    codeword = code.encode(message)

    # SMCS binds (log_height, width), then the structure hash pins the jagged
    # chip layout into the commitment.
    commitment, digest_layers = smcs.commit(codeword, column_major=True)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    # The retained codeword is the open's Merkle-query input, in the [N, K]
    # leaf-major layout ``committed_codeword`` produces — so transpose only here
    # and only when kept. ``drop_codeword`` (SP1's drop_ldes) omits the ~6 GB
    # blow-up from the @jit outputs (XLA then frees the transpose entirely
    # instead of pinning it device-resident); the open re-encodes it via
    # ``committed_codeword`` (fractalyze/sp1-zorch#55, #124).
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
