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

import frx
import frx.numpy as fnp
from frx import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import (
    BasefoldCommitment,
    BasefoldConfig,
    BasefoldProof,
    CadenceProof,
)
from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.pcs.fold import lane_combine, open_rows, to_base_field
from zorch.poly.multilinear import eval_mle
from zorch.prove import fold_rounds
from zorch.round import ProverRound
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver
    from zorch.round import ProverRound


@partial(
    frx.tree_util.register_dataclass,
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


# (codeword, opaque kernel round state, level) — the kernel owns the state's
# shape (native: running MLE + claim + unbound z suffix); `level` counts the
# round so the choreography's per-round grind schedule can key on it.
_OpenCarry = tuple[Array, tuple, int]

# One round's collected artifacts: the raw sumcheck message components (proof
# wire), the pre-fold commit root (proof wire), the committed pair-leaves +
# digest layers (query phase), and this round's grind witness (None unless
# scheduled).
_RoundMsg = tuple[tuple, Array, Array, list[Array], "Array | None"]


@dataclass(frozen=True)
class _SumcheckPairFoldRound(ProverRound):
    """One interleaved-sumcheck round of the batch open, driven by the kernel +
    choreography. Ask the kernel for the round-message components, frame them
    (`round_message`) and emit (`observe_message`), commit the pre-fold
    conjugate-pair leaves and observe the root through `observe_root` (decoupled
    from the fold — unlike `PreFoldPairCommitRound`, which couples
    commit+observe+fold — so a consumer can reframe the root, e.g. a
    truncated-root hash), grind if the schedule says so, sample the shared
    challenge β, then fold the codeword by `code.fold` and the sumcheck state by
    the kernel using that same β. The default kernel+choreography reproduce the
    native wire: `observe(msg) → observe(root) → sample(β)` with a pass-through
    message and root. msg = (message components, root, pre-fold leaves, digest
    layers, grind witness)."""

    code: FoldableCode
    tree: MerkleTree
    choreography: BasefoldChoreography
    kernel: SumcheckKernel

    def __call__(
        self, carry: _OpenCarry, transcript: Transcript
    ) -> tuple[_OpenCarry, Transcript, _RoundMsg]:
        cw, state, level = carry
        chor = self.choreography
        components = self.kernel.message(state)
        msg = chor.round_message(*components)
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
        state = self.kernel.fold(state, components, beta)
        return (
            (cw, state, level + 1),
            t,
            (components, root, leaves, digest_layers, witness),
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
    kernel: SumcheckKernel = SumcheckKernel()
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
        self._resolved_config(num_vars).require_native("open")
        return _open_batch_body(self, list(rounds), z, transcript)

    def open_with_basis(
        self,
        prover_data: BasefoldProverData,
        basis: Array,
        value: Array,
        transcript: Transcript,
    ) -> tuple[BasefoldProof | CadenceProof, Transcript]:
        """Open the batched claim `<f, basis> = value` for a RAW hypercube basis
        instead of a point — the entry of outer protocols whose eval-claims
        arrive as an already-batched basis vector. Mirrors
        `LigeritoProver.open_with_basis`: no point exists, so the choreography's
        `bind_statement` receives `point=None` and must bind the statement
        another way (the native binding refuses — this entry is for a basis
        consumer).

        Under a non-native fold schedule (`row_batch_prefix` / `fold_arities`)
        this returns a generic `CadenceProof` the consumer serializes; under the
        native uniform schedule it returns a `BasefoldProof` (and the native
        binding refuses the basis entry, as before)."""
        if basis.shape[0] != prover_data.mle.shape[0]:
            raise ValueError(
                f"basis length {basis.shape[0]} must equal the MLE height "
                f"{prover_data.mle.shape[0]} (= 2^num_vars)"
            )
        num_vars = log2_strict_usize(basis.shape[0])
        config = self._resolved_config(num_vars)
        if not config.commits_per_round:
            # Non-native fold schedule (row-batch prefix + multi-arity epochs):
            # the eager, host-Fiat-Shamir cadence driver, returning a generic
            # `CadenceProof` the consumer serializes. A byte-wire consumer drives
            # this entry with its own choreography + kernel.
            return _open_with_basis_cadence(
                self, prover_data, basis, value, config, transcript
            )
        return _open_with_basis_body(self, prover_data, basis, value, transcript)


def _open_with_basis_cadence(
    prover: BasefoldProver,
    pd: BasefoldProverData,
    basis: Array,
    value: Array,
    config: BasefoldConfig,
    transcript: Transcript,
) -> tuple[CadenceProof, Transcript]:
    """Eager driver for a non-native fold schedule: the interleaved sumcheck
    (kernel) + a row-batch prefix (deferred lane-combine, one commit at prefix
    end) + multi-arity epoch commits (post-fold, next-arity leaf grouping), all
    FS-framed by the choreography. Mirrors `LigeritoProver._open`'s eager
    orchestration (jitted kernels inside), NOT the jitted native basefold bodies:
    a host-sequential byte-wire transcript can't ride one jit zone.

    `pd.codeword` is the flat codeword the fold walks (interleave lanes grouped
    per position); `pd.mle` seeds the kernel's sumcheck state alongside `basis`.
    Returns a `CadenceProof` — the raw per-layer openings + sumcheck artifacts —
    for the consumer to serialize into its wire (roots, octopus, query tuples)."""
    chor = prover.choreography
    kernel = prover.kernel
    code = prover.code
    tree = prover.tree
    num_vars = config.num_vars
    prefix = config.row_batch_prefix
    arities = config.fold_arities
    num_epochs = len(arities)
    n_pos = code.block_len
    num_ntts = 1 << prefix
    # Fail loud on a scheduled grind BEFORE producing any bytes, symmetric to the
    # native path's refusal: `CadenceProof.pow_witnesses` is the frozen wire slot,
    # but the cadence open does not populate it yet (grind production is a deferred
    # delta). The schedule count comes off the choreography's bits methods
    # (`num_pow_witnesses`) so this and the native/verifier guards share one
    # source of truth.
    if chor.num_pow_witnesses(config) > 0:
        raise NotImplementedError(
            "the choreography schedules a grind, but the non-native cadence open "
            "does not populate CadenceProof.pow_witnesses yet (the field is the "
            "wire slot; the grind production is a deferred delta, symmetric to the "
            "native path)"
        )

    # Initial commit (leaf = the interleave lanes of one position). Re-derived
    # here for the query openings only — the outer protocol already bound this
    # root, so the choreography does NOT observe it now (its `bind_statement`
    # binds whatever the wire binds, e.g. a domain label). The cadence commits
    # the leaves directly, NOT through `to_base_field` like the native path: the
    # consumer owns the leaf grouping (the interleave-lane layout it committed),
    # so routing through `to_base_field` here would change its wire.
    init_leaves = pd.codeword.reshape(n_pos, num_ntts)
    init_root, init_digest = _commit_leaves(tree, init_leaves)
    t = chor.bind_statement(transcript, init_root, None, value)

    # Layers opened in the query phase, in commit order: layer 0 the initial
    # commit at the full index, then one per committed layer; the shift sequence
    # (`config.layer_shifts()`) maps a query position to each layer's folded leaf
    # index (position >> shift).
    layer_leaves: list[Array] = [init_leaves]
    layer_digests: list[list[Array]] = [init_digest]
    commit_roots: list[Array] = []

    state = kernel.initial_state(pd.mle, basis, value)
    cw = None if prefix > 0 else pd.codeword.reshape(n_pos)
    round_messages: list[tuple] = []
    rb_challenges: list[Array] = []
    epoch = in_epoch = 0

    for rnd in range(num_vars):
        components = kernel.message(state)
        msg = chor.round_message(*components)
        t = chor.observe_message(t, msg)
        t, r = chor.fold_challenge(t, None, rnd, 0)
        state = kernel.fold(state, components, r)
        round_messages.append(components)

        if rnd < prefix:
            rb_challenges.append(r)
            if rnd + 1 == prefix:
                cw = lane_combine(init_leaves, rb_challenges)  # [n_pos]
                # No FRI epochs (log_dim == 0): the row-batched codeword is the
                # terminal, so there is no post-prefix commit to make (latent
                # until a log_dim==0 consumer, but correct by construction).
                if arities:
                    lf = 1 << arities[0]
                    leaves = cw.reshape(n_pos // lf, lf)
                    root, digest = _commit_leaves(tree, leaves)
                    t = chor.observe_root(t, root)
                    commit_roots.append(root)
                    layer_leaves.append(leaves)
                    layer_digests.append(digest)
        else:
            cw = code.fold(cw, r)
            in_epoch += 1
            if in_epoch == arities[epoch]:
                if epoch + 1 < num_epochs:
                    next_arity = arities[epoch + 1]
                    lf = 1 << next_arity
                    leaves = cw.reshape(cw.shape[0] // lf, lf)
                    root, digest = _commit_leaves(tree, leaves)
                    t = chor.observe_root(t, root)
                    commit_roots.append(root)
                    layer_leaves.append(leaves)
                    layer_digests.append(digest)
                in_epoch = 0
                epoch += 1

    final_codeword = cw
    final_state = kernel.final(state)
    # Bind the terminal (the native wire observes the whole codeword; a consumer
    # that binds nothing here overrides `observe_final` to a no-op).
    t = chor.observe_final(t, final_codeword)

    # Query phase: shared positions; open every committed layer at its shifted
    # leaf index. `open_rows` returns generic `Opening`s the consumer converts.
    t, positions = chor.sample_queries(t, n_pos, config.num_queries)
    layer_openings: list = []
    layer_positions: list[Array] = []
    layer_num_leaves: list[int] = []
    for leaves, digest, shift in zip(
        layer_leaves, layer_digests, config.layer_shifts()
    ):
        idx = positions >> shift
        layer_openings.append(open_rows(tree, leaves, digest, idx))
        layer_positions.append(idx)
        layer_num_leaves.append(int(leaves.shape[0]))

    proof = CadenceProof(
        round_messages=round_messages,
        commit_roots=commit_roots,
        final_codeword=final_codeword,
        final_state=final_state,
        layer_openings=layer_openings,
        layer_positions=layer_positions,
        layer_num_leaves=layer_num_leaves,
        positions=positions,
    )
    return proof, t


# Jitted cadence-layer commit island: the eager cadence driver
# (`_open_with_basis_cadence`) keeps its host-sequential Fiat-Shamir
# orchestration eager, but each Merkle commit is pure compute that must lower to
# one fused kernel (`tree.commit` → one fused_region), so it rides its own jit
# island — mirroring the native `_commit_body` and ligerito's `_commit`. `tree`
# is the static key (frozen, value-compared, #214).
@partial(frx.jit, static_argnames=("tree",))
def _commit_leaves(tree: MerkleTree, leaves: Array) -> tuple[Array, list[Array]]:
    return tree.commit(leaves)


# Jitted commit body: standalone (outside the jagged seam's enclosing jit), an
# eager commit dispatches the per-column encode ffts and the Merkle
# fused_region op-by-op; inside an enclosing jit it traces straight through.
# Keyed on code + tree, not the prover: commit never reads num_queries, so
# provers differing only there must not compile twice (static keys compare by
# value — #214).
@partial(frx.jit, static_argnames=("code", "tree"))
def _commit_body(
    code: FoldableCode, tree: MerkleTree, polys: list[Array]
) -> tuple[BasefoldCommitment, BasefoldProverData]:
    # The columns share one message length S, so the whole matrix encodes as
    # one [K, S] batch — a single NTT kernel (the LinearCode seam batches over
    # leading axes) — transposed into the [n, K] row-leaf layout the Merkle
    # commit expects.
    mle = fnp.stack(polys, axis=1)
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
    kernel = prover.kernel
    num_vars = config.num_vars

    # Fail loud on a scheduled grind BEFORE the fold loop (not after): the fold
    # round produces a pow witness, but the native driver does not yet collect it
    # into `BasefoldProof.pow_witnesses` (the field is the frozen wire slot; the
    # grind production is a deferred delta), so refuse without wasted fold work
    # rather than drop witnesses silently. The count comes off the choreography's
    # bits methods (`num_pow_witnesses`), the one source of truth shared with the
    # cadence + verifier guards.
    if chor.num_pow_witnesses(config) > 0:
        raise NotImplementedError(
            "scheduled grind produces pow witnesses, but the native open does not "
            "collect them into BasefoldProof.pow_witnesses yet (the field is the "
            "wire slot; the grind production is a deferred delta)"
        )

    # Interleaved sumcheck + pre-fold pair-leaf FRI fold, num_vars rounds. Every
    # round commits its pre-fold layer; the final folded codeword (length
    # `blowup`) is the cleartext final poly. The kernel owns the round state's
    # shape (native: the MLE + running claim + unbound point suffix).
    state = kernel.initial_state(mle, zs, claim)
    carry: _OpenCarry = (cw, state, 0)
    carry, t, msgs = fold_rounds(
        _SumcheckPairFoldRound(prover.code, prover.tree, chor, kernel),
        carry,
        transcript,
        num_vars,
    )
    final_poly = carry[0]
    uni_msgs = [m[0] for m in msgs]
    fri_roots = [m[1] for m in msgs]
    layer_leaves = [m[2] for m in msgs]
    layer_digests = [m[3] for m in msgs]

    # Bind the cleartext final codeword before sampling queries, so the query
    # positions depend on it (the IOPP terminal binding; `verify` mirrors).
    t = chor.observe_final(t, final_poly)

    # Query phase: shared positions; open every matrix at the full index and
    # every fold layer's pair-leaf at its halved index. No grind: the eager guard
    # above refused any scheduled grind before the fold loop.
    n = prover.code.block_len
    t, positions = chor.sample_queries(t, n, config.num_queries)
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
@partial(frx.jit, static_argnames=("prover",))
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
    t = t.observe(fnp.asarray(num_vars, dtype))

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
@partial(frx.jit, static_argnames=("prover",))
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
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[BasefoldCommitment, BasefoldProverData, BasefoldProof]
    ] = BasefoldProver
    _fold_round: type[ProverRound] = _SumcheckPairFoldRound
