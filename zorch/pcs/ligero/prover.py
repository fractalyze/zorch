# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Single-shot Ligero prover — the `PcsProver` half of the matrix-commitment PCS.

`commit` lays one multilinear `f` out as a `rows x cols` matrix `X̃`
(`rows = code.message_len`, `cols = len(f) / rows`), low-degree-extends each
column (`LinearCode.encode`; the native-NTT Reed-Solomon today) and Merkle-commits
the codeword rows — structurally basefold's matrix commit, but the columns are the
low-bit slices of one polynomial rather than a batch of separate ones.

`open` at `z = (z_row, z_col)` sends `w = X̃ · r_col` (`r_col = eq(z_col)`) in the
clear and opens a few codeword rows. Unlike basefold there is no fold-to-end: the
verifier checks proximity `<X[s], r_col> = encode(w)[s]` on the opened rows and
value `<r_row, w> = y` directly. Ligero needs only
`LinearCode.encode`, not the `FoldableCode` fold seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.linear_code import LinearCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.fold import open_rows, sample_positions
from zorch.pcs.ligero.config import LigeroCommitment, LigeroProof
from zorch.pcs.matrix_commit import commit_matrix
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.transcript import Transcript
from zorch.utils.bits import is_power_of_two, log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["root", "matrix", "leaves", "digest_layers"],
    meta_fields=["cols"],
)
@dataclass(frozen=True)
class LigeroProverData:
    """Retained witness from `LigeroProver.commit`: the commitment root, the
    message-domain matrix `X̃` `[rows, cols]` (open dots it with `r_col`), the
    committed base-field leaves (what the Merkle commit/open hashes), the Merkle
    digest layers, plus `cols`. A pytree so `commit`/`open` ride a `@jit` zone."""

    root: Array
    matrix: Array  # [rows, cols] message-domain X̃
    leaves: Array  # [block_len, cols*limbs] base-field Merkle leaves
    digest_layers: list[Array]
    cols: int


@dataclass(frozen=True)
class LigeroProver:
    """Single-shot Ligero PCS prover (`PcsProver`). `code` fixes the matrix row
    count (`= message_len`); `tree` commits the codeword rows."""

    code: LinearCode
    tree: MerkleTree
    num_queries: int = 4  # query repetitions; placeholder, not soundness-calibrated

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[LigeroCommitment, LigeroProverData]:
        """Commit one multilinear laid out as a `rows x cols` matrix. The seam
        takes a `Sequence[Array]`; single-shot Ligero commits exactly one poly
        (batching several is a follow-up)."""
        if len(polys) != 1:
            raise ValueError(
                f"single-shot Ligero commits exactly one polynomial, got {len(polys)}"
            )
        f = polys[0]
        rows = self.code.message_len
        if f.ndim != 1 or f.shape[0] % rows != 0 or not is_power_of_two(f.shape[0]):
            raise ValueError(
                f"polynomial length {f.shape[0]} must be a power of two and a "
                f"multiple of rows={rows} (= code.message_len)"
            )
        cols = f.shape[0] // rows
        return _commit_body(self.code, self.tree, f, rows, cols)

    def open(
        self,
        prover_data: LigeroProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, LigeroProof, Transcript]:
        """Open the committed matrix at the shared point `z`. Returns
        `(value, proof, transcript)` with `value` the scalar `f(z)`."""
        if len(points) != 1:
            raise ValueError(f"Ligero opens at one point, got {len(points)}")
        z = points[0]
        num_vars = z.shape[0]
        k_row = log2_strict_usize(self.code.message_len)
        if num_vars < k_row:
            raise ValueError(
                f"point dimension {num_vars} is fewer than the row variables "
                f"{k_row} (= log2 message_len)"
            )
        if (1 << (num_vars - k_row)) != prover_data.cols:
            raise ValueError(
                f"point dimension {num_vars} doesn't match matrix cols "
                f"{prover_data.cols} (expected 2^(num_vars - {k_row}))"
            )
        return _open_body(self, prover_data, z, transcript)


# Jitted commit body, keyed on code + tree by value (#214): commit never reads
# num_queries, so provers differing only there must not compile twice. `rows` and
# `cols` are static (shape-determining), passed positionally after the arrays.
@partial(jax.jit, static_argnames=("code", "tree", "rows", "cols"))
def _commit_body(
    code: LinearCode, tree: MerkleTree, f: Array, rows: int, cols: int
) -> tuple[LigeroCommitment, LigeroProverData]:
    # X̃[i, j] = f[i*cols + j]: row index = high k_row bits, col index = low
    # k_col bits (lexicographic), so z = (z_row, z_col) splits MSB-first and
    # f(z) = r_row @ X̃ @ r_col.
    matrix = f.reshape(rows, cols)
    # Commit the columns (each length rows): `commit_matrix` encodes along the
    # message axis, so pass `matrix.T` `[cols, rows]` as the [batch, message_len].
    cm, _ = commit_matrix(code, tree, matrix.T)
    return cm.root, LigeroProverData(
        root=cm.root,
        matrix=matrix,
        leaves=cm.leaves,
        digest_layers=cm.digest_layers,
        cols=cols,
    )


# Jitted open body: the prover is the static key (by value, #214).
@partial(jax.jit, static_argnames=("prover",))
def _open_body(
    prover: LigeroProver,
    pd: LigeroProverData,
    z: Array,
    transcript: Transcript,
) -> tuple[Array, LigeroProof, Transcript]:
    dtype = z.dtype
    one = jnp.ones((), dtype)
    k_row = log2_strict_usize(prover.code.message_len)
    z_row, z_col = z[:k_row], z[k_row:]
    r_row = expand_eq_to_hypercube(z_row, one)  # (rows,)
    r_col = expand_eq_to_hypercube(z_col, one)  # (cols,)

    # Broadcast-multiply + sum, not `@`: field matmul is not the zorch idiom
    # (eval_mle contracts the same way) and extension dtypes reject some ops.
    w = (pd.matrix * r_col[None, :]).sum(axis=1)  # (rows,)  = X̃ · r_col
    value = (r_row * w).sum()  # scalar = f(z)

    # FS commit step (mirror `verify`): bind the root, the opened value, then the
    # sent vector w — so the query positions depend on all three.
    t = transcript.observe(pd.root)
    t = t.observe(value)
    t = t.observe(w)

    n = prover.code.block_len
    t, positions = sample_positions(t, n, prover.num_queries)
    opening = open_rows(prover.tree, pd.leaves, pd.digest_layers, positions)
    return value, LigeroProof(w=w, component_opening=opening), t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[PcsProver[LigeroCommitment, LigeroProverData, LigeroProof]] = (
        LigeroProver
    )
