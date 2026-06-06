# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fold-phase machinery shared by the Merkle-committed folding opens (fri,
basefold): the commit-and-fold prover round, the query-opening pair type, the
query-phase assembly, and the Fiat-Shamir position/index derivations both
sides of each scheme use.

These are scheme-neutral — fri and basefold fold the same way (sample β →
`code.fold` → commit the half-size layer → observe the root) and run the same
query phase over the committed layers — so they live at the `pcs` level rather
than under either scheme's package. The pair layout inside each layer is the
code's identity, so the query phase reads it off the `FoldableCode` seam
(`layer_positions` / `pair_indices`) instead of assuming an order. The
position derivation is shared so prover and verifier sample identical query
indices from the transcript, the way the sumcheck block shares one
module-level oracle to keep the two sides in lockstep.

The round loop stays a Python `for` via `zorch.prove.fold_rounds`: each round
Merkle-commits a half-size layer whose retained artifacts are ragged across
rounds, so it is not `lax.scan`-shaped (docs/conventions.md "Loops")."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree, Opening
from zorch.round import Round
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.round import ProverRound


@partial(jax.tree_util.register_dataclass, data_fields=["lo", "hi"], meta_fields=[])
@dataclass(frozen=True)
class LayerOpening:
    """A committed fold layer's opened point pair, batched over the queries
    (leading axis = query count) so the whole query phase is one device op.
    The pair's leaf indices are the code's `pair_indices(a[layer], layer)`.

    A registered pytree so proofs carrying it cross a `@jit` boundary."""

    lo: Opening  # the pair's lo leg (batched over queries)
    hi: Opening  # the pair's hi leg (batched over queries)


@dataclass(frozen=True)
class CommittedLayer:
    """One commit-and-fold round's retained artifacts: the fold challenge (a
    composing round may fold its own state by the same β), the root (proof
    wire), and the committed half-size layer + digest layers (query phase)."""

    beta: Array
    root: Array
    matrix: Array  # [half_block_len, 1]
    digest_layers: list[Array]


@dataclass(frozen=True)
class CommitFoldRound(Round):
    """The shared commit-and-fold prover round: sample β → `code.fold` →
    `tree.commit` the half-size layer → observe the root. Carry = the codeword;
    msg = `CommittedLayer`.

    Run the first R−1 rounds through `fold_rounds` and peel the final round
    explicitly — it observes the folded layer in the clear instead of
    committing it. A scheme with extra per-round state (basefold's interleaved
    sumcheck) wraps this round and folds that state by the msg's β."""

    code: FoldableCode
    tree: MerkleTree

    def __call__(
        self, cw: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, CommittedLayer]:
        t, beta = transcript.sample()
        beta = beta.reshape(())
        cw = self.code.fold(cw, beta)
        matrix = cw.reshape(-1, 1)
        root, digest_layers = self.tree.commit(matrix)
        t = t.observe(root)
        return cw, t, CommittedLayer(beta, root, matrix, digest_layers)


def open_query_phase(
    code: FoldableCode,
    tree: MerkleTree,
    transcript: Transcript,
    base_matrix: Array,
    base_digest_layers: list[Array],
    layers: list[CommittedLayer],
    num_queries: int,
) -> tuple[Transcript, LayerOpening, list[LayerOpening]]:
    """The query phase shared by the fri/basefold `open`: sample the query
    positions, then open the point pair of the base matrix (layer 0) and of
    each committed fold layer, with the pair layout read off `code`. All on
    device — positions are an int32 array and each Merkle opening is one
    `vmap` over them, no host indices or per-query loop. `layers` are the
    `fold_rounds` msgs; the peeled final round is cleartext in the proof, so
    it needs no opening."""
    n = base_matrix.shape[0]
    num_rounds = len(layers) + 1  # committed layers + the peeled final round

    t, positions = sample_positions(transcript, n, num_queries)
    a = code.layer_positions(positions, num_rounds)

    def open_pair(matrix: Array, dl: list[Array], layer: int) -> LayerOpening:
        lo_idx, hi_idx = code.pair_indices(a[layer], layer)

        def open_batch(indices: Array) -> Opening:
            return jax.vmap(lambda i: tree.open(matrix, dl, i))(indices)

        return LayerOpening(open_batch(lo_idx), open_batch(hi_idx))

    base = open_pair(base_matrix, base_digest_layers, 0)
    layer_opens = [
        open_pair(committed.matrix, committed.digest_layers, i)
        for i, committed in enumerate(layers, start=1)
    ]
    return t, base, layer_opens


def sample_positions(
    transcript: Transcript, block_len: int, count: int
) -> tuple[Transcript, Array]:
    """Squeeze `count` query positions in `[0, block_len)` as one device int32
    array — no host round-trip — derived identically on both sides. Each squeezed
    field element's low limb is reduced mod `block_len`."""
    t, raw = transcript.sample(count)
    limbs = lax.bitcast_convert_type(raw, jnp.uint32).reshape(count, -1)
    return t, (limbs[:, 0] % block_len).astype(jnp.int32)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[ProverRound] = CommitFoldRound
