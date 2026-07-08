# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold prover — the commit slice of the multilinear PCS on the `pcs` seam.

`commit` is the low-degree extension of each column (`FoldableCode.encode`; the
native-NTT Reed-Solomon today) followed by a Merkle commit of the codeword rows.
Unlike `kzg`/`fri` — which commit each polynomial in the batch independently and
return one root per poly — BaseFold is a **matrix commitment**: the columns share
one code domain and the Merkle leaves are codeword *rows* spanning all columns, so
the whole batch binds under a single root.

`open` is the BaseFold batch open: it reduces one or more *separately committed*
matrices, evaluated at a shared point, to a single FRI. Each matrix's columns are
combined with the others' by a staggered partial-Lagrange RLC into one codeword
and one MLE; an interleaved sumcheck folds that MLE while the FRI folds the
codeword by the same per-round challenge, the round committing the *pre-fold*
layer's conjugate-pair leaves before sampling the fold challenge. A single matrix
is the degenerate one-round batch.

The open is driven by a `(BasefoldConfig, BasefoldChoreography)` pair: the config
fixes the fold schedule (commit cadence + leaf grouping), the choreography fixes
the Fiat-Shamir framing (round-message form, root/terminal observes, query
sampling, grinding). `BasefoldProver`'s defaults are zorch's native wire —
`commits_per_round` (pre-fold arity-2 pair commit every round) + the native
`BasefoldChoreography` — so the plain `BasefoldProver(code, tree, num_queries=…)`
construction is byte-for-byte today's implementation. A byte-fixed consumer
supplies its own config + choreography and drives `open_with_basis` instead —
the raw-basis entry mirroring `LigeritoProver.open_with_basis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldCommitment, BasefoldConfig, BasefoldProof
from zorch.pcs.fold import open_rows, to_base_field
from zorch.poly.multilinear import eval_mle, mle_fold
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver
    from zorch.round import ProverRound


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["digest_layers", "mle", "codeword", "leaves"],
    meta_fields=["widths"],
)
@dataclass(frozen=True)
class BasefoldProverData:
    """Retained witness from `BasefoldProver.commit`: the Merkle digest layers,
    the message-domain MLE `[S, K]` (the sumcheck folds it), the codeword
    `[block_len, K]` (the fold halves it), the committed base-field leaves (what
    the Merkle commit/open hashes — the codeword's rows reinterpreted as
    base-field limbs, identity for a base-field code), plus per-column widths. A
    pytree so `commit`/`open` ride a `@jit` zone."""

    digest_layers: list[Array]
    mle: Array  # [S, K] message-domain columns
    codeword: Array  # [block_len, K] codeword (the fold halves it)
    leaves: Array  # [block_len, K*limbs] base-field Merkle leaves
    widths: tuple[int, ...]


def _sumcheck_msg(mle: Array, claim: Array, zs: Array) -> tuple[Array, Array]:
    """The degree-1 sumcheck message `(s(0), s(1))` for the variable bound this
    round (`zs[-1]`), from the running MLE and claim. `mle_fold(., 0)` fixes the
    bound variable to 0 (the additive fold coincides with the multilinear
    partial-eval at beta=0), so zero_val is the sumcheck s(0); one_val is
    recovered from the running claim. The zorch-native (point-driven) message
    form — the basis entry's `(u0, u2)` product form is a consumer delta."""
    zero_mle = mle_fold(mle, jnp.zeros((), zs.dtype))
    rest = zs[:-1]
    zero_val = eval_mle(zero_mle, rest) if rest.shape[0] > 0 else zero_mle[0]
    one_val = (claim - zero_val) / zs[-1] + zero_val
    return zero_val, one_val


# (codeword, running MLE, running claim, unbound z suffix | None, level) — the
# suffix shrinks with the MLE, `level` counts the round so the choreography's
# per-round grind schedule can key on it. `zs is None` is the raw-basis entry
# (`open_with_basis`), where no point exists.
_OpenCarry = tuple[Array, Array, Array, "Array | None", int]

# One round's collected artifacts: the sumcheck message pieces, the pre-fold
# commit root (proof wire), the committed pair-leaves + digest layers (query
# phase), and this round's grind witness (None unless scheduled).
_RoundMsg = tuple[tuple[Array, Array], Array, Array, list[Array], "Array | None"]


@dataclass(frozen=True)
class _SumcheckPairFoldRound(Round):
    """One interleaved-sumcheck round of the batch open, driven by the
    choreography. Emit the round message (`round_message` + `observe_message`),
    commit the pre-fold conjugate-pair leaves and observe the root through
    `observe_root` (decoupled from the fold — unlike `PreFoldPairCommitRound`,
    which couples commit+observe+fold — so a consumer can reframe the root, e.g.
    a truncated-root hash), grind if the schedule says so, sample the shared
    challenge β, then fold the codeword *and* the MLE and reduce the running
    claim by that same β. The default choreography reproduces the native wire:
    `observe(msg) → observe(root) → sample(β)` with a pass-through message and
    root. msg = (sumcheck message, root, pre-fold leaves, digest layers,
    grind witness)."""

    code: FoldableCode
    tree: MerkleTree
    choreography: BasefoldChoreography

    def __call__(
        self, carry: _OpenCarry, transcript: Transcript
    ) -> tuple[_OpenCarry, Transcript, _RoundMsg]:
        cw, mle, claim, zs, level = carry
        chor = self.choreography
        if zs is None:
            # The raw-basis entry (`open_with_basis`) has no point, so the
            # native point-driven message cannot be formed. The basis wire's
            # message (a product form over (mle, basis)) is a consumer delta,
            # wired with that consumer; the native path always has a point.
            raise NotImplementedError(
                "open_with_basis's basis-path message is not wired here: the "
                "native driver forms (s(0), s(1)) from the opening point; a "
                "basis consumer supplies its own message form from (mle, basis)"
            )
        zero_val, one_val = _sumcheck_msg(mle, claim, zs)
        msg = chor.round_message(zero_val, one_val)
        t = chor.observe_message(transcript, msg)
        # Pre-fold pair commit, decoupled so the root observe routes through the
        # choreography (native `observe_root` is a pass-through `observe`).
        leaves = to_base_field(self.code.pair_leaves(cw))
        root, digest_layers = self.tree.commit(leaves)
        t = chor.observe_root(t, root)
        # Per-round grind, ground between the message absorb and the challenge
        # squeeze (native schedule: None -> nothing on the wire, no advance).
        bits = chor.fold_grind_bits(level, 0)
        witness: Array | None = None
        if bits is not None:
            t, witness = chor.grind(t, bits)
        t, beta = t.sample()
        beta = beta.reshape(())
        cw = self.code.fold(cw, beta)
        mle = mle_fold(mle, beta)
        claim = zero_val + beta * one_val
        return (
            (cw, mle, claim, zs[:-1], level + 1),
            t,
            ((zero_val, one_val), root, leaves, digest_layers, witness),
        )


@dataclass(frozen=True)
class BasefoldProver:
    """BaseFold PCS prover (`PcsProver`). `code` fixes the per-column message
    length (= the MLE height `S`); `tree` commits the codeword rows;
    `choreography` fixes the Fiat-Shamir wire (share the instance with the
    verifier); `config` fixes the fold schedule. The defaults are zorch's native
    wire — the plain `BasefoldProver(code, tree, num_queries=…)` construction is
    byte-for-byte today's implementation. `config=None` derives the native
    per-open config (`commits_per_round`, `num_vars` from the opening point)."""

    code: FoldableCode
    tree: MerkleTree
    num_queries: int = 4  # query repetitions; placeholder, not soundness-calibrated
    choreography: BasefoldChoreography = BasefoldChoreography()
    config: BasefoldConfig | None = None

    def _resolved_config(self, num_vars: int) -> BasefoldConfig:
        """The config driving an open over `num_vars` variables: the explicit
        one (checked against the point dimension) or the native default
        (`commits_per_round`, `num_queries` from the prover)."""
        if self.config is None:
            return BasefoldConfig(num_vars=num_vars, num_queries=self.num_queries)
        if self.config.num_vars != num_vars:
            raise ValueError(
                f"config.num_vars={self.config.num_vars} doesn't match the open's "
                f"variable count {num_vars}"
            )
        return self.config

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[BasefoldCommitment, BasefoldProverData]:
        return _commit_body(self.code, self.tree, list(polys))

    def open(
        self,
        prover_data: BasefoldProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, BasefoldProof, Transcript]:
        """Open a single committed matrix — the degenerate one-round batch, the
        `PcsProver` seam shape. Returns `(values, proof, transcript)` with
        `values` the matrix's per-column evaluations `[K]`."""
        values, proof, t = self.open_batch([prover_data], points, transcript)
        return values[0], proof, t

    def open_batch(
        self,
        rounds: Sequence[BasefoldProverData],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[list[Array], BasefoldProof, Transcript]:
        """Batch-open the committed matrices `rounds` at the shared point.

        Returns `(values, proof, transcript)` where `values[r]` is round `r`'s
        per-column evaluations `[w_r]`. A single-element `rounds` is the
        degenerate (un-batched) open.
        """
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrices at one shared point, got {len(points)}"
            )
        if not rounds:
            raise ValueError("BaseFold opens at least one committed matrix, got none")
        z = points[0]  # (log_S,)
        num_vars = z.shape[0]
        # Eager shape guards, ahead of the jit zone (mirrors the verifier).
        if num_vars < 1:
            raise ValueError("BaseFold opens over at least one variable, got none")
        for pd in rounds:
            if pd.mle.shape[0] != (1 << num_vars):
                raise ValueError(
                    f"point dimension {num_vars} doesn't match MLE height "
                    f"{pd.mle.shape[0]} (expected 2^{num_vars})"
                )
        _require_native_cadence(self._resolved_config(num_vars))
        return _open_batch_body(self, list(rounds), z, transcript)

    def open_with_basis(
        self,
        prover_data: BasefoldProverData,
        basis: Array,
        value: Array,
        transcript: Transcript,
    ) -> tuple[BasefoldProof, Transcript]:
        """Open the batched claim `<f, basis> = value` for a RAW hypercube basis
        instead of a point — the entry of outer protocols whose eval-claims
        arrive as an already-batched basis vector. Mirrors
        `LigeritoProver.open_with_basis`: no point exists, so the choreography's
        `bind_statement` receives `point=None` and must bind the statement
        another way (the native binding refuses — this entry is for a basis
        consumer)."""
        if basis.shape[0] != prover_data.mle.shape[0]:
            raise ValueError(
                f"basis length {basis.shape[0]} must equal the MLE height "
                f"{prover_data.mle.shape[0]} (= 2^num_vars)"
            )
        num_vars = log2_strict_usize(basis.shape[0])
        _require_native_cadence(self._resolved_config(num_vars))
        return _open_with_basis_body(self, prover_data, basis, value, transcript)


def _require_native_cadence(config: BasefoldConfig) -> None:
    """Fail loud on a non-native fold schedule: this driver drives only
    `commits_per_round` (pre-fold arity-2 pair commit every round). The
    row-batch prefix + multi-arity epoch cadence is the deferred fold-schedule
    machinery (design §"Core driver" step 2), wired + byte-gated with its first
    consumer; only the config STRUCTURE for it exists here."""
    if not config.commits_per_round:
        raise NotImplementedError(
            "non-native fold cadence (row_batch_prefix / fold_arities) is not "
            "driven yet; only commits_per_round (zorch-native) is wired. The "
            "deferred row-batch-prefix + multi-arity epoch cadence lands with "
            "its first byte-fixed consumer"
        )


# Jitted commit body: standalone (outside the jagged seam's enclosing jit), an
# eager commit dispatches the per-column encode ffts and the Merkle
# fused_region op-by-op; inside an enclosing jit it traces straight through.
# Keyed on code + tree, not the prover: commit never reads num_queries, so
# provers differing only there must not compile twice (static keys compare by
# value — #214).
@partial(jax.jit, static_argnames=("code", "tree"))
def _commit_body(
    code: FoldableCode, tree: MerkleTree, polys: list[Array]
) -> tuple[BasefoldCommitment, BasefoldProverData]:
    # The columns share one message length S, so the whole matrix encodes as
    # one [K, S] batch — a single NTT kernel (the LinearCode seam batches over
    # leading axes) — transposed into the [n, K] row-leaf layout the Merkle
    # commit expects.
    mle = jnp.stack(polys, axis=1)
    codeword = code.encode(mle.T).T
    leaves = to_base_field(codeword)
    root, layers = tree.commit(leaves)
    return root, BasefoldProverData(
        digest_layers=layers,
        mle=mle,
        codeword=codeword,
        leaves=leaves,
        widths=(len(polys),),
    )


def _fold_and_query(
    prover: BasefoldProver,
    config: BasefoldConfig,
    cw: Array,
    mle: Array,
    claim: Array,
    zs: Array | None,
    component_pds: Sequence[BasefoldProverData],
    transcript: Transcript,
) -> tuple[BasefoldProof, Transcript]:
    """The config+choreography-driven core, shared by the point (`open_batch`)
    and raw-basis (`open_with_basis`) entries: run the interleaved sumcheck +
    pre-fold FRI (num_vars rounds), bind the cleartext terminal codeword, then
    the query phase. Assumes `config.commits_per_round` (the caller guards).

    `cw` / `mle` / `claim` are the already-combined single codeword / MLE /
    claim; `zs` is the opening-point suffix (None under the basis entry);
    `component_pds` are the committed matrices opened at the query positions."""
    chor = prover.choreography
    num_vars = config.num_vars

    # Interleaved sumcheck + pre-fold pair-leaf FRI fold, num_vars rounds. Every
    # round commits its pre-fold layer; the final folded codeword (length
    # `blowup`) is the cleartext final poly.
    carry: _OpenCarry = (cw, mle, claim, zs, 0)
    carry, t, msgs = fold_rounds(
        _SumcheckPairFoldRound(prover.code, prover.tree, chor),
        carry,
        transcript,
        num_vars,
    )
    final_poly = carry[0]
    uni_msgs = [m[0] for m in msgs]
    fri_roots = [m[1] for m in msgs]
    layer_leaves = [m[2] for m in msgs]
    layer_digests = [m[3] for m in msgs]
    pow_witnesses = [m[4] for m in msgs if m[4] is not None]

    # Bind the cleartext final codeword before sampling queries, so the query
    # positions depend on it (the IOPP terminal binding; `verify` mirrors).
    t = chor.observe_final(t, final_poly)

    # Query phase: shared positions; open every matrix at the full index and
    # every fold layer's pair-leaf at its halved index.
    n = prover.code.block_len
    qbits = chor.query_grind_bits(0)
    if qbits is not None:
        t, witness = chor.grind(t, qbits)
        pow_witnesses.append(witness)
    t, positions = chor.sample_queries(t, n, config.num_queries)
    if pow_witnesses:
        # A grinding schedule produced witnesses, but `BasefoldProof` has no pow
        # field to carry them (the native wire grinds nothing). The grind wire —
        # a grinding consumer's delta — lands with the field in a later adoption.
        raise NotImplementedError(
            "scheduled grind produced pow witnesses, but BasefoldProof carries "
            "no pow field yet (the native wire grinds nothing); the grind wire "
            "is a deferred consumer delta"
        )
    a = prover.code.layer_positions(positions, num_vars)
    component_openings = [
        open_rows(prover.tree, pd.leaves, pd.digest_layers, positions)
        for pd in component_pds
    ]
    query_openings = [
        open_rows(prover.tree, layer_leaves[i], layer_digests[i], a[i])
        for i in range(num_vars)
    ]
    proof = BasefoldProof(
        uni_msgs, fri_roots, final_poly, component_openings, query_openings
    )
    return proof, t


# Jitted open body: an eager replay re-traces the per-round pair-leaf
# `open_rows` vmaps per call (issue #186); unlike `commit` — which the jagged
# seam also reaches inside its enclosing jit — `open` is reached eagerly via
# `stacked_open`. The prover is the static key (by value, #214), so its config
# and choreography (both frozen, value-compared) fix the compiled zone.
@partial(jax.jit, static_argnames=("prover",))
def _open_batch_body(
    prover: BasefoldProver,
    rounds: Sequence[BasefoldProverData],
    z: Array,
    transcript: Transcript,
) -> tuple[list[Array], BasefoldProof, Transcript]:
    dtype = z.dtype
    num_vars = z.shape[0]
    config = prover._resolved_config(num_vars)
    t = transcript
    # 1. Bind every commitment root into the transcript (the FS commit step,
    #    mirroring `fri`). `verify` observes the same roots in the same order.
    #    This multi-matrix statement binding is the native (zorch) consumer's
    #    staggered-RLC batching — a `bind_statement` peer, kept here rather than
    #    on the choreography because "combine separate matrices" is where
    #    consumers genuinely diverge (some have none, using `open_with_basis`).
    for pd in rounds:
        t = t.observe(pd.digest_layers[-1][0])

    # 2. Per-matrix, per-column evaluations at z; observe each as sampled.
    values = []
    for pd in rounds:
        vals = eval_mle(pd.mle, z, axis=0)  # (w_r,)
        t = t.observe(vals)
        values.append(vals)

    # 3. Staggered partial-Lagrange RLC over the total column width, then
    #    collapse every matrix's columns into one MLE / codeword / claim.
    total_width = sum(int(pd.mle.shape[1]) for pd in rounds)
    t, coeffs = sample_staggered_coeffs(t, total_width, dtype)
    current_mle = batch_staggered([pd.mle for pd in rounds], coeffs)  # (S,)
    cw = batch_staggered([pd.codeword for pd in rounds], coeffs)  # (n,)
    current_claim = batch_staggered(values, coeffs)  # scalar
    # Domain separation: bind the fold-round count (mirrors the reference).
    t = t.observe(jnp.asarray(num_vars, dtype))

    # 4-7. Config+choreography-driven fold + terminal binding + query phase.
    proof, t = _fold_and_query(
        prover, config, cw, current_mle, current_claim, z, rounds, t
    )
    return values, proof, t


# Jitted raw-basis open body (the `open_with_basis` entry): binds the statement
# through the choreography with `point=None`, then drives the shared core. The
# native choreography refuses `point=None` (this entry is for a basis consumer);
# the basis-path sumcheck message is a deferred consumer delta (see
# `_SumcheckPairFoldRound`).
@partial(jax.jit, static_argnames=("prover",))
def _open_with_basis_body(
    prover: BasefoldProver,
    pd: BasefoldProverData,
    basis: Array,
    value: Array,
    transcript: Transcript,
) -> tuple[BasefoldProof, Transcript]:
    num_vars = log2_strict_usize(basis.shape[0])
    config = prover._resolved_config(num_vars)
    root = pd.digest_layers[-1][0]
    # Statement binding via the choreography — no point, so `bind_statement`
    # gets `point=None` (native refuses; a basis consumer binds via the basis).
    t = prover.choreography.bind_statement(transcript, root, None, value)
    # The single committed matrix's codeword / MLE drive the fold; the basis
    # takes the point's place in the message form, wired by a basis consumer.
    # The reduction never runs on the native choreography (bind_statement raised
    # above); the basis-path column combination + message form land with that
    # consumer.
    cw = pd.codeword[:, 0]
    mle = pd.mle[:, 0]
    del basis  # consumed by the deferred basis-path message form
    return _fold_and_query(prover, config, cw, mle, value, None, [pd], t)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[BasefoldCommitment, BasefoldProverData, BasefoldProof]
    ] = BasefoldProver
    _fold_round: type[ProverRound] = _SumcheckPairFoldRound
