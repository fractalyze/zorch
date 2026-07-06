# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito recursive-open prover — the `PcsProver` half of the recursive matrix
PCS. Single-shot Ligero (`pcs/ligero`) sends the vector `w = X̃·r_col` in the
clear, paying `sqrt(N)`. Ligerito instead *commits* `w` and discharges the
proximity check as one continuous interleaved sumcheck that batches every level's
committed-`w` eval-claims and recurses on the residual.

The whole thing is one sumcheck `Σ_x W(x)·B(x) = claim` over the shrinking witness
`W` (`W` starts as the committed multilinear `f`, `B` as the value basis `eq(z)`,
`claim` as `y = f(z)`). Each level:

  * folds `fold_ks[j]` variables (the current committed matrix `M_j`'s interleave
    lanes) through the degree-2 product sumcheck;
  * re-commits the folded `W` as a fresh Ligero matrix `M_{j+1}` at a lower rate;
  * opens `M_j`'s codeword rows and *induces* its proximity eval-claims
    `<M_j[s], eq(c_j)> = eval_mle(W_folded, eval_point(s))`
    into the running sumcheck with a fresh separation challenge — the
    `induce_sumcheck_poly` analog, built code-generically from the `TensorCode`
    seam's `eval_point` rather than a basis-specific novel-basis tensor.

The final level sends the folded residual in the clear; its proximity ties the
residual to the last committed matrix directly (no sumcheck needed), and the
sumcheck's terminal claim closes against `Σ_x residual(x)·B(x)`.

The commit encodes `mle_evals_to_coeffs(matrix)` (not the matrix directly): that
coeff transform cancels the `TensorCode` seam's `coeffs_to_evals`, so both the
proximity RHS and the value check read as clean `eval_mle`s of the *eval-basis*
witness — the whole recursion stays in one basis (design note: the seam identity
`encode(w)[s] == eval_mle(coeffs_to_evals(w), eval_point(s))`).

Reuses `pcs/basefold`'s staggered partial-Lagrange batching for the per-level
`α` weights and `pcs/fold`'s query machinery (`sample_positions` / `open_rows`).
Code-generic over a `TensorCode`; the multiplicative Reed-Solomon instantiation
is the de-risk vehicle (fractalyze/flock-zorch#32).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.tensor_code import TensorCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.basefold.batching import sample_staggered_coeffs
from zorch.pcs.fold import open_rows, sample_positions
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.matrix_commit import CommittedMatrix, commit_matrix
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import mle_evals_to_coeffs
from zorch.sumcheck.prover import SumcheckRound
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize
from zorch.utils.field import field_sum

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver

# A code factory: (message_len, log_inv_rate) -> the level's TensorCode. Keeping
# the code per-level (rate shrinks each level) and code-generic — RS is the
# de-risk instantiation, additive-NTT is #11/#27.
MakeCode = Callable[[int, int], TensorCode]

# The degree-2 product round (`W·B`) drives every sumcheck round; one instance,
# reused. The claim reduces through `eval_univariate` (value-form message).
_ROUND = SumcheckRound(degree=2)


# Jitted commit body, keyed on code + tree + interleave by value (#214): commit
# never reads the query count, so provers differing only in `queries` reuse this
# compiled function rather than re-tracing.
@partial(jax.jit, static_argnames=("log_interleave", "code", "tree"))
def _commit(
    witness: Array, log_interleave: int, code: TensorCode, tree: MerkleTree
) -> CommittedMatrix:
    """Ligero-commit the eval-basis `witness` as a matrix whose interleave lanes
    are its high `log_interleave` variables and whose message (encoded) axis is
    the low ones. Encodes `mle_evals_to_coeffs(matrix)` so the codeword coordinate
    at position `s` is a clean `eval_mle(., eval_point(s))` of the eval-basis
    message (the seam's `coeffs_to_evals` is cancelled by this `evals_to_coeffs`)."""
    kappa = 1 << log_interleave
    rho = witness.shape[0] // kappa
    if code.message_len != rho:
        raise ValueError(
            f"code.message_len={code.message_len} must equal the message length "
            f"{rho} (= 2^(vars - interleave))"
        )
    matrix = witness.reshape(kappa, rho)
    cm, _ = commit_matrix(code, tree, matrix, pre=mle_evals_to_coeffs)
    return cm


@dataclass(frozen=True)
class LigeritoProverData:
    """Retained witness from `LigeritoProver.commit`: the eval-basis multilinear
    `f` and the initial matrix commitment `M_0`. `open` runs the recursion off
    these."""

    f: Array
    initial: CommittedMatrix


@dataclass(frozen=True)
class LigeritoProver:
    """Ligerito recursive PCS prover. `make_code(message_len, log_inv_rate)`
    builds each level's `TensorCode`; `config` fixes the fold schedule; `tree`
    commits every level's codeword rows."""

    make_code: MakeCode
    tree: MerkleTree
    config: LigeritoConfig

    def _code(self, level: int, message_len: int) -> TensorCode:
        return self.make_code(message_len, self.config.log_inv_rates[level])

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[LigeritoCommitment, LigeritoProverData]:
        """Commit one multilinear as the initial Ligero matrix `M_0` (interleave =
        `fold_ks[0]` lanes)."""
        if len(polys) != 1:
            raise ValueError(
                f"Ligerito commits exactly one polynomial, got {len(polys)}"
            )
        f = polys[0]
        num_vars = log2_strict_usize(f.shape[0])
        if num_vars != self.config.num_vars:
            raise ValueError(
                f"polynomial has {num_vars} variables, config expects "
                f"{self.config.num_vars} (= sum(fold_ks))"
            )
        k0 = self.config.fold_ks[0]
        code0 = self._code(0, 1 << (num_vars - k0))
        initial = _commit(f, k0, code0, self.tree)
        return initial.root, LigeritoProverData(f=f, initial=initial)

    def open(
        self,
        prover_data: LigeritoProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, LigeritoProof, Transcript]:
        """Open the committed multilinear at `z`, returning `(value, proof,
        transcript)` with `value = f(z)`."""
        if len(points) != 1:
            raise ValueError(f"Ligerito opens at one point, got {len(points)}")
        z = points[0]
        if z.shape[0] != self.config.num_vars:
            raise ValueError(
                f"point dimension {z.shape[0]} must equal the variable count "
                f"{self.config.num_vars}"
            )
        return _open(self, prover_data, z, transcript)


def _open(
    prover: LigeritoProver,
    pd: LigeritoProverData,
    z: Array,
    transcript: Transcript,
) -> tuple[Array, LigeritoProof, Transcript]:
    cfg = prover.config
    dtype = z.dtype
    one = jnp.ones((), dtype)

    # The continuous sumcheck state: witness W (folds) and basis B (glued +
    # folds). B starts as the value basis eq(z). The prover only emits round
    # messages and folds — it never tracks the running claim (the verifier
    # reduces and checks it); `value = f(z)` is the opened value it returns.
    W = pd.f
    B = expand_eq_to_hypercube(z, one)
    value = field_sum(pd.f * B)  # f(z) = <f, eq(z)>; reuse B rather than rebuild eq(z)

    # Bind the statement (root, point, value) before any challenge.
    t = transcript.observe(pd.initial.root)
    t = t.observe(z)
    t = t.observe(value)

    sumcheck_messages: list[Array] = []
    recursive_roots: list[Array] = []
    component_openings: list[Opening] = []

    current = pd.initial  # M_j
    num_vars = cfg.num_vars
    for j in range(cfg.num_levels):
        k_j = cfg.fold_ks[j]
        # --- fold this level's k_j lane variables through the product sumcheck ---
        for _ in range(k_j):
            [W, B], t, msg = _ROUND([W, B], t)
            sumcheck_messages.append(msg)
        num_vars -= k_j

        # --- re-commit the folded witness as M_{j+1} (non-final levels) ---
        is_final = j == cfg.num_levels - 1
        if not is_final:
            k_next = cfg.fold_ks[j + 1]
            code_next = prover._code(j + 1, 1 << (num_vars - k_next))
            nxt = _commit(W, k_next, code_next, prover.tree)
            t = t.observe(nxt.root)
            recursive_roots.append(nxt.root)
        else:
            # Bind the in-clear residual before sampling the final level's queries
            # (the IOPP terminal binding — queries depend on it; verify mirrors).
            t = t.observe(W)

        # --- open M_j's codeword rows at the sampled query positions ---
        # M_j's message (encoded) axis is exactly the post-fold witness, so its
        # message length is 2^num_vars — rebuild the same code the commit used.
        code_j = prover._code(j, 1 << num_vars)
        t, positions = sample_positions(t, code_j.block_len, cfg.queries[j])
        opening = open_rows(
            prover.tree, current.leaves, current.digest_layers, positions
        )
        component_openings.append(opening)

        if is_final:
            # The final level's proximity ties the in-clear residual to M_j and
            # is checked directly by the verifier — no induce/glue here.
            break

        # Induce: glue this level's proximity eval-point basis into the running
        # sumcheck. B_new(x) = Σ_s α_s·eq(eval_point(s), x); the α weights and the
        # separation challenge match what the verifier resamples, so it rebuilds
        # the same B. Only B is threaded — the enforced sum is verifier-side.
        t, alpha = sample_staggered_coeffs(t, cfg.queries[j], dtype)
        alpha = alpha[: cfg.queries[j]]  # (Q,) partial-Lagrange weights
        points_s = code_j.eval_point(positions)  # (Q, num_vars) message-var points
        eqps = jax.vmap(lambda p: expand_eq_to_hypercube(p, one))(points_s)  # (Q, 2^nv)
        b_new = field_sum(alpha[:, None] * eqps, axis=0)  # (2^num_vars,)
        t, sep = t.sample()
        sep = sep.reshape(())
        B = B + sep * b_new
        current = nxt

    residual = W  # the final folded witness, sent in the clear
    proof = LigeritoProof(
        sumcheck_messages=sumcheck_messages,
        recursive_roots=recursive_roots,
        component_openings=component_openings,
        final_residual=residual,
        ood_values=[],
    )
    return value, proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[LigeritoCommitment, LigeritoProverData, LigeritoProof]
    ] = LigeritoProver
