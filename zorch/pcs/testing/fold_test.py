# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The k-group fold orchestration: a self-contained k-ary low-degree test.

Exercises the additive k-ary machinery in `pcs/fold` — `PreFoldKGroupCommitRound`,
`KGroupCommittedLayer`, and `verify_group_fold_chain` — as a plain LDT (no DEEP
quotient): the layer-0 commitment is the FRI polynomial itself, so the prover
just commits-and-folds it by `fold_factor` each round and the verifier replays
the transcript, rebuilds the query positions, and checks the opened k-groups fold
down the committed chain. This is the k-ary mirror of `fri_test`'s round-trip on
the scheme-neutral fold seam, without binding to any one scheme.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.fold import (
    PreFoldKGroupCommitRound,
    open_rows,
    sample_positions,
    verify_group_fold_chain,
    verify_openings,
)
from zorch.prove import fold_rounds
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


def _transcript() -> Transcript:
    # Prover and verifier each build a fresh, identical sponge, so both draw the
    # same Fiat-Shamir stream (fold challenges then query positions).
    return cheap_transcript(KB)


class KGroupFoldRoundTripTest(absltest.TestCase):
    """A degree-<16 polynomial folded by factor 4 twice lands on a length-2
    constant codeword: two k-group commit rounds, then a query phase over the
    committed chain. message_len 16 * blowup 2 -> block_len 32 -> 8 -> 2."""

    def setUp(self) -> None:
        self.k = 4
        self.num_rounds = 2
        self.num_queries = 3
        _, _, self.tree = koalabear16_merkle()
        self.code = ReedSolomon(16, 2, KB, fold_factor=self.k)
        self.f = self.code.encode(rand_field(1, (16,), KB))

    def _prove(self) -> tuple[Array, list[Array], list]:
        t = _transcript()
        cw, t, layers = fold_rounds(
            PreFoldKGroupCommitRound(self.code, self.tree), self.f, t, self.num_rounds
        )
        final = cw
        t = t.observe(final)
        t, positions = sample_positions(t, self.code.block_len, self.num_queries)
        a = self.code.group_layer_positions(positions, self.num_rounds)
        query_openings = [
            open_rows(self.tree, layer.leaves, layer.digest_layers, a[i])
            for i, layer in enumerate(layers)
        ]
        roots = [layer.root for layer in layers]
        return final, roots, query_openings

    def _verify(self, final: Array, roots: list[Array], query_openings: list) -> Array:
        # The honest verifier owns no prover state: it replays the transcript to
        # recover the fold challenges and query positions, then checks the chain.
        t = _transcript()
        betas = []
        for root in roots:
            t = t.observe(root)
            t, beta = t.sample()
            betas.append(beta.reshape(()))
        t = t.observe(final)
        t, positions = sample_positions(t, self.code.block_len, self.num_queries)
        a = self.code.group_layer_positions(positions, self.num_rounds)

        ok = self.code.check_final(final, final[0])
        legs = [(roots[i], a[i], query_openings[i]) for i in range(self.num_rounds)]
        ok = ok & verify_openings(self.tree, legs)
        ok = ok & verify_group_fold_chain(self.code, query_openings, betas, a, final)
        return ok

    def test_honest_opening_verifies(self) -> None:
        final, roots, query_openings = self._prove()
        self.assertTrue(bool(self._verify(final, roots, query_openings)))

    def test_final_layer_is_constant(self) -> None:
        # Two factor-4 folds drop a degree-<16 poly to degree 0, so the final
        # length-2 layer is a constant codeword.
        final, _, _ = self._prove()
        self.assertEqual(final.shape, (2,))
        self.assertTrue(bool(jnp.all(final == final[0])))

    def test_tampered_final_layer_rejected(self) -> None:
        final, roots, query_openings = self._prove()
        tampered = final.at[1].add(jnp.array(1, dtype=KB))
        self.assertFalse(bool(self._verify(tampered, roots, query_openings)))

    def test_tampered_query_group_rejected(self) -> None:
        # Corrupting an opened k-group breaks both its Merkle rebuild and the
        # fold-chain step that consumes it.
        final, roots, query_openings = self._prove()
        bad = query_openings[0]
        bad = dataclasses.replace(bad, row=bad.row.at[0, 0].add(jnp.array(1, dtype=KB)))
        query_openings = [bad, query_openings[1]]
        self.assertFalse(bool(self._verify(final, roots, query_openings)))

    def test_fold_group_jits(self) -> None:
        # The prover-side k-ary fold is a jitted kernel (Option B): it must lower
        # under @jax.jit and match the eager result.
        beta = rand_field(60, (), KB)
        jitted = jax.jit(self.code.fold_group)
        self.assertTrue(
            bool(jnp.all(jitted(self.f, beta) == self.code.fold_group(self.f, beta)))
        )

    def test_verify_group_fold_chain_jits(self) -> None:
        # The verifier fold-chain check is a jitted kernel: the openings are
        # pytrees and the round count is static, so the whole check rides one jit
        # (the k-ary counterpart of the binary verifier's @jax.jit body).
        final, roots, query_openings = self._prove()
        t = _transcript()
        betas = []
        for root in roots:
            t = t.observe(root)
            t, beta = t.sample()
            betas.append(beta.reshape(()))
        t = t.observe(final)
        t, positions = sample_positions(t, self.code.block_len, self.num_queries)
        a = self.code.group_layer_positions(positions, self.num_rounds)
        jitted = jax.jit(
            lambda qo, b, idx, fp: verify_group_fold_chain(self.code, qo, b, idx, fp)
        )
        self.assertTrue(bool(jitted(query_openings, betas, a, final)))


class KGroupFoldBinaryFactorTest(KGroupFoldRoundTripTest):
    """fold_factor 2 drives the same orchestration as the k=2 case — reusing the
    round-trip harness proves the k-ary path is a strict generalization that also
    folds binary (via the Lagrange seam). block_len 8 -> 4 -> 2, so the final
    layer is again length 2 and every inherited round-trip case holds."""

    def setUp(self) -> None:
        self.k = 2
        self.num_rounds = 2
        self.num_queries = 3
        _, _, self.tree = koalabear16_merkle()
        self.code = ReedSolomon(4, 2, KB, fold_factor=2)  # block_len 8 -> 4 -> 2
        self.f = self.code.encode(rand_field(2, (4,), KB))


if __name__ == "__main__":
    absltest.main()
