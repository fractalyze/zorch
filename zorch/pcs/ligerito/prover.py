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

The commit basis — the `(pre, expand)` pair fixing how a codeword coordinate
reads back as a point-eval of the committed row — is a `CommitBasis`
(`ligerito/basis.py`), selected by `LigeritoConfig.monomial_commit`. The default
`EVAL_BASIS` encodes `mle_evals_to_coeffs(matrix)`, cancelling the `TensorCode`
seam's `coeffs_to_evals` so both the proximity RHS and the value check read as
clean `eval_mle`s of the *eval-basis* witness (`encode(w)[s] ==
eval_mle(coeffs_to_evals(w), eval_point(s))`); `MONOMIAL_BASIS` is the raw-lane
(coefficient) convention a byte-fixed consumer needs (flock, #32).

Reuses `pcs/basefold`'s staggered partial-Lagrange batching for the per-level
`α` weights and `pcs/fold`'s query machinery (`open_rows`). Every transcript
interaction routes through the `LigeritoChoreography` seam (statement binding, round
hops, root/residual framing, query sampling), so a byte-fixed consumer swaps
the wire without touching the recursion. Code-generic over a `TensorCode`; the
multiplicative Reed-Solomon instantiation is the de-risk vehicle
(fractalyze/flock-zorch#32).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as jnp
from frx import Array

from zorch.coding.tensor_code import TensorCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.basefold.batching import sample_staggered_coeffs
from zorch.pcs.fold import open_rows
from zorch.pcs.ligerito.basis import EVAL_BASIS, CommitBasis, select_commit_basis
from zorch.pcs.ligerito.choreography import LigeritoChoreography
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.matrix_commit import CommittedMatrix, commit_matrix
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.domain import fold
from zorch.sumcheck.prover import (
    CompressedProductRound,
    ProductSummand,
    StandardRound,
)
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver

# A code factory: (message_len, log_inv_rate) -> the level's TensorCode. Keeping
# the code per-level (rate shrinks each level) and code-generic — RS is the
# de-risk instantiation, additive-NTT is #11/#27.
MakeCode = Callable[[int, int], TensorCode]

# The degree-2 product round (`W·B`) drives every sumcheck round; one instance,
# reused. The claim reduces through `eval_univariate` (value-form message).
_ROUND = StandardRound(ProductSummand(degree=2))
_COMPRESSED_ROUND = CompressedProductRound()


# Jitted commit body, keyed on code + tree + interleave by value (#214): commit
# never reads the query count, so provers differing only in `queries` reuse this
# compiled function rather than re-tracing. `basis` is static too — its `.pre` is
# the encode pre-transform (the singletons are identity-stable keys).
@partial(frx.jit, static_argnames=("log_interleave", "code", "tree", "basis"))
def _commit(
    witness: Array,
    log_interleave: int,
    code: TensorCode,
    tree: MerkleTree,
    basis: CommitBasis = EVAL_BASIS,
) -> CommittedMatrix:
    """Ligero-commit the `witness` as a matrix whose interleave lanes are its high
    `log_interleave` variables and whose message (encoded) axis is the low ones,
    encoding `basis.pre(matrix)` so a codeword coordinate reads back as
    `<row, basis.expand(eval_point(s))>` (see `zorch.pcs.ligerito.basis`). Reads
    only `basis.pre`; the paired `basis.expand` is the open/verify side."""
    kappa = 1 << log_interleave
    rho = witness.shape[0] // kappa
    if code.message_len != rho:
        raise ValueError(
            f"code.message_len={code.message_len} must equal the message length "
            f"{rho} (= 2^(vars - interleave))"
        )
    matrix = witness.reshape(kappa, rho)
    cm, _ = commit_matrix(code, tree, matrix, pre=basis.pre)
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
    commits every level's codeword rows; `choreography` fixes the Fiat-Shamir
    wire (share the instance with the verifier)."""

    make_code: MakeCode
    tree: MerkleTree
    config: LigeritoConfig
    choreography: LigeritoChoreography = LigeritoChoreography()

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
        basis = select_commit_basis(self.config.monomial_commit)
        initial = _commit(f, k0, code0, self.tree, basis=basis)
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
        one = jnp.ones((), z.dtype)
        B = expand_eq_to_hypercube(z, one)
        value = (prover_data.f * B).sum()  # f(z) = <f, eq(z)>; reuse B
        proof, t = _open(self, prover_data, z, B, value, transcript)
        return value, proof, t

    def open_with_basis(
        self,
        prover_data: LigeritoProverData,
        basis: Array,
        value: Array,
        transcript: Transcript,
    ) -> tuple[LigeritoProof, Transcript]:
        """Open the batched claim `<f, basis> = value` for a RAW hypercube basis
        instead of a point — the entry of outer protocols whose eval-claims
        arrive as an already-batched basis vector (flock's
        `recursive_prover_with_basis`). No point exists, so the choreography's
        `bind_statement` receives `point=None` and must bind the statement
        another way (the native binding refuses)."""
        if basis.shape[0] != 1 << self.config.num_vars:
            raise ValueError(
                f"basis length {basis.shape[0]} must be 2^{self.config.num_vars}"
            )
        return _open(self, prover_data, None, basis, value, transcript)


def _open(
    prover: LigeritoProver,
    pd: LigeritoProverData,
    z: Array | None,
    B: Array,
    value: Array,
    transcript: Transcript,
) -> tuple[LigeritoProof, Transcript]:
    # Static-config recursion → the whole open compiles to ONE device program;
    # eager, the tiny per-round ops dispatch from Python and starve the GPU.
    return _open_jit(prover, pd.f, pd.initial, z, B, value, transcript)


@partial(frx.jit, static_argnums=(0,))
def _open_jit(
    prover: LigeritoProver,
    W0: Array,
    initial: CommittedMatrix,
    z: Array | None,
    B: Array,
    value: Array,
    transcript: Transcript,
) -> tuple[LigeritoProof, Transcript]:
    cfg = prover.config
    chor = prover.choreography
    dtype = B.dtype
    one = jnp.ones((), dtype)
    round_ = _COMPRESSED_ROUND if cfg.compressed_sumcheck_messages else _ROUND
    basis = select_commit_basis(cfg.monomial_commit)

    # The continuous sumcheck state: witness W (folds) and basis B (glued +
    # folds). B starts as the statement's basis — eq(z) under `open`, the raw
    # vector under `open_with_basis`. The prover only emits round messages and
    # folds — it never tracks the running claim (the verifier reduces and
    # checks it).
    W = W0

    # Bind the statement before any challenge (z is None under the basis entry).
    t = chor.bind_statement(transcript, initial.root, z, value)

    sumcheck_messages: list[Array] = []
    recursive_roots: list[Array] = []
    component_openings: list[Opening] = []
    component_positions: list[Array] = []
    ood_values: list[Array] = []
    pow_witnesses: list[Array] = []

    # Under the eager policy each state's message is emitted (absorbed + put on
    # the wire) the moment the state forms; under the lazy default the round
    # message rides `fold_challenge`'s fused hop instead. The round poly is a
    # full pass over the state, so the policy gates WHICH point computes it —
    # never both.
    eager = chor.eager_messages

    def emit(t: Transcript, witness: Array, basis: Array) -> Transcript:
        msg = round_._round_poly(jnp.stack([witness, basis]))
        sumcheck_messages.append(msg)
        return chor.observe_message(t, msg)

    def grind(t: Transcript, bits: int | None) -> Transcript:
        if bits is None:
            return t
        t, witness = chor.grind(t, bits)
        pow_witnesses.append(witness)
        return t

    if eager:
        t = emit(t, W, B)

    current = initial  # M_j
    num_vars = cfg.num_vars
    for j in range(cfg.num_levels):
        k_j = cfg.fold_ks[j]
        # --- fold this level's k_j lane variables through the product sumcheck ---
        for i in range(k_j):
            msg: Array | None = None  # eager: this round's is already absorbed
            if not eager:
                msg = round_._round_poly(jnp.stack([W, B]))
                sumcheck_messages.append(msg)
            t = grind(t, chor.fold_grind_bits(j, i))
            t, r = chor.fold_challenge(t, msg, j, i)
            W, B = fold(jnp.stack([W, B]), r)
            if eager:
                # The freshly folded state's — the terminal residual state's
                # included (the verifier recomputes that one in the clear).
                t = emit(t, W, B)
        num_vars -= k_j

        # --- re-commit the folded witness as M_{j+1} (non-final levels) ---
        is_final = j == cfg.num_levels - 1
        if not is_final:
            k_next = cfg.fold_ks[j + 1]
            code_next = prover._code(j + 1, 1 << (num_vars - k_next))
            nxt = _commit(W, k_next, code_next, prover.tree, basis=basis)
            t = chor.observe_root(t, nxt.root)
            recursive_roots.append(nxt.root)
            # --- OOD binding of the fresh commit: glue the eval-claim at a
            # sampled out-of-domain point into the running sumcheck, exactly
            # like induce but with a transcript-drawn basis and a claimed
            # (observed) value. The claim side is verifier-tracked. ---
            for _ in range(cfg.ood_count(j)):
                t, zs = t.sample(num_vars)
                b_ood = expand_eq_to_hypercube(zs.astype(dtype), one)
                y = (W * b_ood).sum()
                t = t.observe(y)
                ood_values.append(y)
                if eager:
                    t = emit(t, W, b_ood)
                t, sep = t.sample()
                sep = sep.reshape(())
                B = B + sep * b_ood
        else:
            # Bind the in-clear residual before sampling the final level's queries
            # (the IOPP terminal binding — queries depend on it; verify mirrors).
            t = chor.observe_residual(t, W)

        # --- open M_j's codeword rows at the sampled query positions ---
        # M_j's message (encoded) axis is exactly the post-fold witness, so its
        # message length is 2^num_vars — rebuild the same code the commit used.
        code_j = prover._code(j, 1 << num_vars)
        t = grind(t, chor.query_grind_bits(j))
        t, positions = chor.sample_queries(t, code_j.block_len, cfg.queries[j])
        opening = open_rows(
            prover.tree, current.leaves, current.digest_layers, positions
        )
        component_openings.append(opening)
        component_positions.append(positions)

        if is_final:
            # The final level's proximity ties the in-clear residual to M_j and
            # is checked directly by the verifier — no induce/glue here.
            break

        # Induce: glue this level's proximity eval-point basis into the running
        # sumcheck. B_new(x) = Σ_s α_s·eq(eval_point(s), x); the α weights and the
        # separation challenge match what the verifier resamples, so it rebuilds
        # the same B. Only B is threaded — the enforced sum is verifier-side.
        t, alpha = sample_staggered_coeffs(
            t, cfg.queries[j], dtype, lsb_first=cfg.alpha_lsb_first
        )
        alpha = alpha[: cfg.queries[j]]  # (Q,) partial-Lagrange weights
        points_s = code_j.eval_point(positions)  # (Q, num_vars) message-var points
        eqps = basis.proximity_basis(points_s, one)  # (Q, 2^nv)
        b_new = (alpha[:, None] * eqps).sum(axis=0)  # (2^num_vars,)
        if eager:
            t = emit(t, W, b_new)
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
        ood_values=ood_values,
        pow_witnesses=pow_witnesses,
        component_positions=component_positions,
    )
    return proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[LigeritoCommitment, LigeritoProverData, LigeritoProof]
    ] = LigeritoProver
