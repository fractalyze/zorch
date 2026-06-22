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
rounds, so it is not `lax.scan`-shaped (docs/conventions.md "Loops")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode, KFoldableCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.round import Round
from zorch.transcript import Transcript

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
    return jax.vmap(lambda i: tree.open(matrix, digest_layers, i))(indices)


def sample_positions(
    transcript: Transcript, block_len: int, count: int
) -> tuple[Transcript, Array]:
    """Squeeze `count` query positions in `[0, block_len)` as one device int32
    array — no host round-trip — derived identically on both sides. Each squeezed
    field element's low limb is reduced mod `block_len`."""
    t, raw = transcript.sample(count)
    limbs = lax.bitcast_convert_type(raw, jnp.uint32).reshape(count, -1)
    return t, (limbs[:, 0] % block_len).astype(jnp.int32)


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

    The loop stays unrolled: `fold_values` is one `lax.fft` plus a few field ops
    per level, so it already traces O(1) per round — scanning it would add a
    control-flow boundary for no trace+lower win (docs/conventions.md "Loops")."""
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
    boundary for no trace+lower win (docs/conventions.md "Loops")."""
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
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = PreFoldPairCommitRound
    _k: type[ProverRound] = PreFoldKGroupCommitRound
