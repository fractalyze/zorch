# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR prover — the `PcsProver` half of the multilinear PCS.

`commit` Reed-Solomon-encodes the polynomial's hypercube evaluations and binds the
codeword rows under a query-strided Merkle root. `open` runs the WHIR rounds: each
round folds the MLE by `k_whir` sumcheck sub-folds, re-encodes the folded MLE as a
fresh (shrinking) RS codeword, commits it, answers one out-of-domain point, and
opens the previous round's codeword at strided query cosets — accumulating the
out-of-domain and in-domain constraints into the weight polynomial. The last round
sends the folded coefficients in the clear.

Unlike fri/basefold this never folds a codeword in place, so it does not use
`pcs/fold.py`; the round driver is WHIR's own. See `config.py` for the divergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.pcs.whir.config import WhirCommitment, WhirParams, WhirProof
from zorch.poly.multilinear import mle_evals_to_coeffs
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["mle", "codeword", "digest_layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class WhirProverData:
    """Retained witness from `WhirProver.commit`: the message-domain MLE the
    sumcheck folds, the initial RS codeword (its strided rows are the Merkle
    leaves the round-0 queries open), and the codeword's Merkle digest layers.
    A pytree so `commit`/`open` ride a `@jit` zone."""

    mle: Array
    codeword: Array
    digest_layers: list[Array]


@dataclass(frozen=True)
class WhirProver:
    """WHIR PCS prover (`PcsProver`). `code` is the initial-round RS encoder (the
    round driver re-encodes at shrinking sizes); `tree` commits the codeword's
    `2^k_whir`-row query cosets; `params` carries the per-round knobs."""

    code: ReedSolomon
    tree: StridedMerkleTree
    params: WhirParams

    def commit(self, polys: Sequence[Array]) -> tuple[WhirCommitment, WhirProverData]:
        """Bind a single multilinear, given by its `2^m` hypercube evaluations.
        Re-expresses it in the coefficient basis, RS-encodes to the initial
        codeword, and strided-Merkle-commits the codeword's `2^k_whir` query
        cosets. Slice 2a commits one polynomial (no μ-batch)."""
        if len(polys) != 1:
            raise ValueError(
                f"WHIR slice 2a commits a single polynomial, got {len(polys)}"
            )
        poly = polys[0]
        if poly.ndim != 1:
            raise ValueError(f"polynomial must be 1-D hypercube evals, got {poly.ndim}")
        if poly.shape[0] != self.code.message_len:
            raise ValueError(
                f"polynomial length {poly.shape[0]} != code message_len "
                f"{self.code.message_len}"
            )
        return _commit_body(self.code, self.tree, poly)

    def open(
        self,
        prover_data: WhirProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, WhirProof, Transcript]:
        raise NotImplementedError("WhirProver.open — Task 5")


# Jitted commit body, keyed on code + tree by value (#214): standalone, an eager
# commit dispatches the encode NTT + Merkle fused_region op-by-op. The MLE is
# re-expressed in the coefficient basis, RS-encoded, and committed as a
# single-column `(block_len, 1)` matrix whose `2^k_whir` strided rows one query
# opens. `mle` (the message-domain evals) and the codeword are retained for `open`.
@partial(jax.jit, static_argnames=("code", "tree"))
def _commit_body(
    code: ReedSolomon, tree: StridedMerkleTree, poly: Array
) -> tuple[WhirCommitment, WhirProverData]:
    codeword = code.encode(mle_evals_to_coeffs(poly))[:, None]
    root, layers = tree.commit(codeword)
    return root, WhirProverData(mle=poly, codeword=codeword, digest_layers=layers)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[PcsProver[WhirCommitment, WhirProverData, WhirProof]] = WhirProver
