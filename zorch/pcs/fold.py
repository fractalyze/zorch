# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fold-phase machinery shared by the Merkle-committed folding opens (fri,
basefold): the pre-fold pair-leaf commit-and-fold prover round, the query-row
opener, and the Fiat-Shamir position derivation both sides use.

These are scheme-neutral — fri and basefold fold the same way (commit the
layer's conjugate-pair leaves → observe the root → sample β → `code.fold`) and
run the same query phase over the committed layers — so they live at the `pcs`
level rather than under either scheme's package. The pair layout inside each
layer is the code's identity, so the round and query phase read it off the
`FoldableCode` seam (`pair_leaves` / `pair_indices` / `layer_positions`) instead
of assuming an order. The position derivation is shared so prover and verifier
sample identical query indices from the transcript.

The round loop stays a Python `for` via `zorch.prove.fold_rounds`: each round
Merkle-commits a half-size layer whose retained artifacts are ragged across
rounds, so it is not `lax.scan`-shaped (docs/reference/conventions.md "Loops")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import frx
import frx.numpy as jnp
import zk_dtypes
from frx import Array, lax

from zorch.coding.foldable_code import FoldableCode, KFoldableCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.round import Round
from zorch.transcript import GrindingTranscript, Transcript, TranscriptT

if TYPE_CHECKING:
    from zorch.round import ProverRound


def to_base_field(leaves: Array) -> Array:
    """A leaf's storage dtype (what the base-field hashers commit) is split from
    its value dtype (what the fold math reads). Reinterpret extension-field
    leaves as base-field limbs, folding the new trailing axis into the leaf
    width. Identity for a base-field code (passes through unchanged)."""
    try:
        bf = zk_dtypes.efinfo(leaves.dtype).base_field_dtype
    except ValueError:
        return leaves
    limbs = lax.bitcast_convert_type(leaves, bf)
    return limbs.reshape(*leaves.shape[:-1], -1)


def from_base_field(rows: Array, dtype: Any, group: int) -> Array:
    """Inverse of `to_base_field` for a `(Q, group * limbs)` opened leaf: split
    the limb axis back out and reinterpret to the code's value dtype, yielding
    `(Q, group)`. Identity for a base-field code (dtype carries no limbs).
    `dtype` can't be inferred — base-field rows have lost which extension field
    they encode — so the caller passes the code's field."""
    try:
        limbs = zk_dtypes.efinfo(dtype).degree
    except ValueError:
        return rows
    return lax.bitcast_convert_type(rows.reshape(rows.shape[0], group, limbs), dtype)


@dataclass(frozen=True)
class PairCommittedLayer:
    """One pre-fold pair-leaf commit round's retained artifacts: the fold
    challenge β (a composing round folds its own state by it), the pair-leaf
    commitment root (proof wire), and the committed `[n//2, 2]` pair-leaves +
    digest layers (query phase). A plain dataclass — a prover-internal
    `fold_rounds` message, not a proof leaf."""

    beta: Array
    root: Array
    leaves: Array  # [n//2, 2] conjugate-pair leaves
    digest_layers: list[Array]


@dataclass(frozen=True)
class PreFoldPairCommitRound(Round):
    """The shared commit-and-fold round, pre-fold pair-leaf schedule: commit the
    codeword's conjugate-pair leaves (`code.pair_leaves`, one leaf = one pair) →
    observe the root → sample β → fold. Committing *before* the fold binds the
    layer into the transcript that samples β — and a pair leaf lets one Merkle
    path open both legs the fold consumes. A scheme with extra per-round state
    (basefold's interleaved sumcheck) wraps this and folds that state by the
    msg's β. Carry = the pre-fold codeword; msg = `PairCommittedLayer`."""

    code: FoldableCode
    tree: MerkleTree

    def __call__(
        self, cw: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, PairCommittedLayer]:
        leaves = to_base_field(self.code.pair_leaves(cw))
        root, digest_layers = self.tree.commit(leaves)
        t = transcript.observe(root)
        t, beta = t.sample()
        beta = beta.reshape(())
        cw = self.code.fold(cw, beta)
        return cw, t, PairCommittedLayer(beta, root, leaves, digest_layers)


@dataclass(frozen=True)
class KGroupCommittedLayer:
    """The k-ary `PairCommittedLayer`: one pre-fold k-group commit round's
    retained artifacts — the fold challenge β, the k-group-leaf commitment root
    (proof wire), and the committed `[n//k, k]` group-leaves + digest layers
    (query phase). `fold_factor` k generalizes the conjugate pair to the k-th-root
    coset; everything else mirrors the binary round."""

    beta: Array
    root: Array
    leaves: Array  # [n//k, k] k-group leaves
    digest_layers: list[Array]


@dataclass(frozen=True)
class PreFoldKGroupCommitRound(Round):
    """The k-ary `PreFoldPairCommitRound`: commit the codeword's k-group leaves
    (`code.group_leaves`, one leaf = one k-th-root coset) → observe the root →
    sample β → fold by `fold_factor`. Committing before the fold binds the layer
    into the transcript that samples β, and a k-group leaf lets one Merkle path
    open all k legs the fold consumes. Carry = the pre-fold codeword; msg =
    `KGroupCommittedLayer`."""

    code: KFoldableCode
    tree: MerkleTree

    def __call__(
        self, cw: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, KGroupCommittedLayer]:
        leaves = to_base_field(self.code.group_leaves(cw))
        root, digest_layers = self.tree.commit(leaves)
        t = transcript.observe(root)
        t, beta = t.sample()
        beta = beta.reshape(())
        cw = self.code.fold_group(cw, beta)
        return cw, t, KGroupCommittedLayer(beta, root, leaves, digest_layers)


def open_rows(
    tree: MerkleTree, matrix: Array, digest_layers: list[Array], indices: Array
) -> Opening:
    """Open `matrix` at every leaf in `indices` as one `vmap`, returning an
    `Opening` whose `row`/`path` carry the query axis. Used for both the
    component matrices (opened at the full query index) and the committed
    fold layers' pair-leaves (opened at the layer's halved index)."""
    return frx.vmap(lambda i: tree.open(matrix, digest_layers, i))(indices)


def lane_combine(lanes: Array, challenges: Sequence[Array]) -> Array:
    """The row-batch prefix's codeword op: fold the trailing lane axis of
    `lanes` `[rows, 2^prefix]` by each prefix challenge in turn (the multilinear
    partial-eval bind `(1-r)·e0 + r·e1`, low bit first), collapsing to `[rows]`.

    Deferred to one pass at prefix end — the lane variables are exactly the ones
    the sumcheck binds over the prefix rounds, so they combine with the same
    challenges. The prover folds the whole post-lane codeword (`[n_pos, 2^prefix]`
    -> `[n_pos]`); the verifier folds each query's opened lanes (`[Q, 2^prefix]`
    -> `[Q]`) — one op, so the two cannot drift. An empty `challenges` (no prefix)
    returns the single lane. Char-2-agnostic: `(1-r)·e0 + r·e1` is the
    field-general bind."""
    buf = lanes
    for r in challenges:
        pairs = buf.reshape(buf.shape[0], -1, 2)
        e0, e1 = pairs[..., 0], pairs[..., 1]
        one = jnp.ones((), e0.dtype)
        buf = (one - r) * e0 + r * e1
    return buf[:, 0]


def sample_positions(
    transcript: TranscriptT, block_len: int, count: int
) -> tuple[TranscriptT, Array]:
    """Squeeze `count` query positions in `[0, block_len)` as one device int32
    array — no host round-trip — derived identically on both sides. Each squeezed
    field element's low limb is reduced mod `block_len`. Generic over the
    transcript flavor so a `GrindingTranscript` caller keeps its type."""
    t, raw = transcript.sample(count)
    limbs = lax.bitcast_convert_type(raw, jnp.uint32).reshape(count, -1)
    return t, (limbs[:, 0] % block_len).astype(jnp.int32)


def sample_distinct_positions(
    transcript: TranscriptT, block_len: int, count: int
) -> tuple[TranscriptT, Array]:
    """Rejection-sample `count` DISTINCT positions in `[0, block_len)`, sorted
    ascending — the strategy of wire formats that spend one squeeze per
    candidate and re-squeeze on a repeat (a distinct-query byte-wire strategy;
    each squeeze reduces like `sample_positions`, low limb mod `block_len`).
    Transcript consumption is data-dependent, so the loop is host-driven (one
    device->host sync per candidate) and cannot run under `jit`; both sides
    replay the identical rejections, so their streams stay in lockstep."""
    if count > block_len:
        raise ValueError(
            f"cannot sample {count} distinct positions from a block of {block_len}"
        )
    seen: set[int] = set()
    out: list[int] = []
    t = transcript
    while len(out) < count:
        t, raw = t.sample(1)
        pos = int(lax.bitcast_convert_type(raw, jnp.uint32).reshape(-1)[0]) % block_len
        if pos not in seen:
            seen.add(pos)
            out.append(pos)
    return t, jnp.sort(jnp.asarray(out, dtype=jnp.int32))


@dataclass(frozen=True)
class FoldChoreography:
    """The Fiat-Shamir choreography shared by the fold-recursion schemes built
    on this module's rounds: the seam that fixes WHEN a recursive open touches
    the transcript, decoupled from WHAT the recursion computes and from
    whatever round algebra a scheme layers on top (its own kernel/config seam).

    Two provers can run the identical recursion (same folds, same commits) and
    still produce different byte streams: one binds the opening point, the
    other binds only the claim; one grinds a proof-of-work between a round
    message and its challenge; one observes each round message the moment it
    forms, the other fuses observe+sample at round start; one derives query
    indices by rejection sampling instead of a plain reduction.
    `FoldChoreography` owns exactly those choices as overridable hooks
    operating on the generic `Transcript`, with zorch's native wire as the
    default behavior — a scheme subclasses this with its own framing/algebra
    hooks, and a byte-fixed consumer subclasses further, overriding only its
    deltas.

    Prover and verifier must share ONE choreography instance: every hook is
    side-neutral (a pure transcript interaction) except the grind/check pair,
    whose schedule both sides read off the same bits methods, so a shared
    instance keeps the two Fiat-Shamir streams equal by construction.

    The message emission policy is the structural choice `eager_messages`
    selects: lazy (default) fuses each round message's absorb with its
    challenge squeeze (`fold_challenge`); eager absorbs a message the moment it
    forms (`observe_message`) and `fold_challenge` samples bare (`msg=None`) —
    the two are one policy, split only so the driver can place the
    interactions."""

    @property
    def eager_messages(self) -> bool:
        """False: round messages ride fused observe+sample hops
        (`fold_challenge`). True: `observe_message` absorbs each message at
        emission time and `fold_challenge` must be overridden to a bare sample
        (its `msg` arrives as None) — the two are one policy, split only so the
        driver can place the interactions."""
        return False

    def bind_statement(
        self, transcript: TranscriptT, root: Array, point: Array | None, value: Array
    ) -> TranscriptT:
        """Bind the opening statement before any challenge. Default binds all
        of (root, point, value) in that order; a consumer whose outer protocol
        already binds the point overrides. `point` is None under a raw-basis
        entry, where no point exists — the native binding refuses rather than
        silently bind less."""
        transcript = transcript.observe(root)
        if point is None:
            raise ValueError(
                "the native statement binding observes the opening point, but "
                "this entry carries none — a basis-entry consumer must "
                "override bind_statement (the basis binds the statement)"
            )
        transcript = transcript.observe(point)
        return transcript.observe(value)

    def observe_message(self, transcript: TranscriptT, msg: Array) -> TranscriptT:
        """Absorb one eagerly emitted message (eager policy only)."""
        return transcript.observe(msg)

    def fold_challenge(
        self, transcript: TranscriptT, msg: Array | None, level: int, fold_idx: int
    ) -> tuple[TranscriptT, Array]:
        """The per-round Fiat-Shamir hop: absorb the round message, squeeze the
        scalar fold challenge. Default is the fused `observe_and_sample` (one
        kernel under `@jit` — the repo's fusion contract). Under the eager
        policy `msg` is None (already absorbed) and the override samples bare."""
        del level, fold_idx  # the default schedule is position-independent
        if msg is None:
            raise ValueError(
                "the lazy default absorbs the round message here; an eager "
                "choreography must override fold_challenge to a bare sample"
            )
        transcript, r = transcript.observe_and_sample(msg, 1)
        return transcript, r[0]

    def observe_root(self, transcript: TranscriptT, root: Array) -> TranscriptT:
        """Absorb a fold round's commit root."""
        return transcript.observe(root)

    def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
        """Proof-of-work schedule for a fold round, ground between the round
        message's absorb and its challenge squeeze. None (default) = no grind
        and nothing on the wire; an int puts a witness on the wire — 0 included
        (a 0-bit grind is trivial but still advances the transcript)."""
        del level, fold_idx
        return None

    def query_grind_bits(self, level: int) -> int | None:
        """Proof-of-work schedule for a level's query phase, ground right
        before its positions are sampled. Same None / int-including-0 contract
        as `fold_grind_bits`."""
        del level
        return None

    def grind(self, transcript: TranscriptT, bits: int) -> tuple[TranscriptT, Array]:
        """Prover-side grind (called only when the bits schedule says so).
        Default is the `GrindingTranscript` seam, so a zorch-native consumer
        adds grinding by overriding only the bits methods; a byte-wire consumer
        overrides the mechanism too."""
        grinding = cast(GrindingTranscript, transcript)
        advanced, witness = grinding.grind(bits)
        return cast(TranscriptT, advanced), witness

    def check_grind(
        self, transcript: TranscriptT, bits: int, witness: Array
    ) -> tuple[TranscriptT, Array]:
        """Verifier-side dual of `grind`: replay the witness, return
        `(transcript, ok)` with the transcript advanced identically."""
        grinding = cast(GrindingTranscript, transcript)
        advanced, ok = grinding.check_witness(bits, witness)
        return cast(TranscriptT, advanced), ok

    def sample_queries(
        self, transcript: TranscriptT, block_len: int, count: int
    ) -> tuple[TranscriptT, Array]:
        """Squeeze `count` query positions in `[0, block_len)`."""
        return sample_positions(transcript, block_len, count)


def verify_openings(
    tree: MerkleTree, legs: Sequence[tuple[Array, Array, Opening]]
) -> Array:
    """AND of "every opened leaf rebuilds its committed root" over a list of
    `(root, indices, opening)` legs — the lo/hi pair of layer 0 and of each
    committed fold layer.

    Legs are grouped by leaf-row width because the leaf hash must see a uniform
    row shape within one `vmap` — the base layer's row is the RLC of all columns,
    a fold layer's is a single column. Each group is rebuilt in one batched
    `tree.reconstruct_roots`, padding the per-layer paths (the tree halves each
    round) to the group's deepest. So the compress body traces once per width
    group instead of once per layer (#163). Shared by the fri and basefold
    verifiers, whose query phase has the same pair-per-layer shape."""
    groups: dict[int, list[tuple[Array, Array, Opening]]] = {}
    for root, idx, opening in legs:
        groups.setdefault(opening.row.shape[-1], []).append((root, idx, opening))

    ok = jnp.bool_(True)
    for group in groups.values():
        max_depth = max(len(opening.path) for _, _, opening in group)
        rows, indices, paths, valid, roots = [], [], [], [], []
        for root, idx, opening in group:
            q = idx.shape[0]
            depth = len(opening.path)
            path = jnp.stack(opening.path, axis=1)  # (queries, depth, digest)
            path = jnp.pad(path, ((0, 0), (0, max_depth - depth), (0, 0)))
            rows.append(opening.row)
            indices.append(idx)
            paths.append(path)
            valid.append(
                jnp.broadcast_to(jnp.arange(max_depth) < depth, (q, max_depth))
            )
            roots.append(jnp.broadcast_to(root, (q, *root.shape)))
        rebuilt = tree.reconstruct_roots(
            jnp.concatenate(rows),
            jnp.concatenate(indices),
            jnp.concatenate(paths),
            jnp.concatenate(valid),
        )
        ok = ok & jnp.all(rebuilt == jnp.concatenate(roots))
    return ok


def verify_fold_chain(
    code: FoldableCode,
    query_openings: Sequence[Opening],
    betas: Sequence[Array],
    leaf_indices: Sequence[Array],
    final_poly: Array,
) -> Array:
    """AND of "each committed fold layer's opened pair folds to the next layer's
    opened value (or the final poly at the last layer)" over every round. The fri
    and basefold verifiers run the same chain once each has tied its layer-0 pair
    to its own source (the DEEP quotient / the staggered component RLC), so the
    per-layer fold lives here on the shared seam. `leaf_indices[i]` is layer `i`'s
    query leaf index `code.layer_positions(positions)[i]`.

    The loop stays unrolled: `fold_values` is one `lax.ntt` plus a few field ops
    per level, so it already traces O(1) per round — scanning it would add a
    control-flow boundary for no trace+lower win
    (docs/reference/conventions.md "Loops")."""
    num_rounds = len(query_openings)
    ok = jnp.bool_(True)
    for i in range(num_rounds):
        leaf = from_base_field(query_openings[i].row, code.dtype, 2)  # (Q, 2)
        folded = code.fold_values(leaf[:, 0], leaf[:, 1], betas[i], leaf_indices[i], i)
        if i < num_rounds - 1:
            # The fold lands at leaf_indices[i] in layer i+1 — the lo or hi leg of
            # that layer's opened pair, decided by the code's layout.
            next_lo, _ = code.pair_indices(leaf_indices[i + 1], i + 1)
            nxt = from_base_field(query_openings[i + 1].row, code.dtype, 2)
            expected = jnp.where(leaf_indices[i] == next_lo, nxt[:, 0], nxt[:, 1])
        else:
            expected = final_poly[leaf_indices[i]]
        ok = ok & jnp.all(folded == expected)
    return ok


def verify_group_fold_chain(
    code: KFoldableCode,
    query_openings: Sequence[Opening],
    betas: Sequence[Array],
    leaf_indices: Sequence[Array],
    final_poly: Array,
) -> Array:
    """The k-ary `verify_fold_chain`: AND of "each committed fold layer's opened
    k-group folds to the next layer's opened value (or the final poly at the last
    layer)" over every round. `query_openings[i].row` is `(Q, k)`, layer `i`'s
    opened k-group at `leaf_indices[i] = code.group_layer_positions(...)[i]`.

    The loop stays unrolled for the same reason as the binary chain:
    `fold_group_values` traces O(1) per round (one `compute_lagrange_basis` plus
    a static-width-k combination), so scanning it would add a control-flow
    boundary for no trace+lower win (docs/reference/conventions.md "Loops")."""
    num_rounds = len(query_openings)
    ok = jnp.bool_(True)
    for i in range(num_rounds):
        group = query_openings[i].row  # (Q, k)
        folded = code.fold_group_values(group, betas[i], leaf_indices[i], i)
        if i < num_rounds - 1:
            # The fold lands at leaf_indices[i] in layer i+1 — one of the k legs
            # of that layer's opened group. Select the leg whose full-layer index
            # equals the landing index (exactly one matches) by the k-way
            # generalization of the binary chain's `jnp.where` on (lo, hi).
            members = code.group_indices(leaf_indices[i + 1], i + 1)
            nxt = query_openings[i + 1].row  # (Q, k)
            expected = jnp.zeros_like(folded)
            for m in range(nxt.shape[-1]):
                expected = jnp.where(members[m] == leaf_indices[i], nxt[:, m], expected)
        else:
            expected = final_poly[leaf_indices[i]]
        ok = ok & jnp.all(folded == expected)
    return ok


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[ProverRound] = PreFoldPairCommitRound
    _k: type[ProverRound] = PreFoldKGroupCommitRound
