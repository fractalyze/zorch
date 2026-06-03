# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BasefoldProver.commit — RS low-degree-extension + Merkle, against an independent
reconstruction of the same codeword + root."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.transcript import StubTranscript


def _basefold(
    log_s: int = 2, blowup: int = 2
) -> tuple[BasefoldProver, ReedSolomon, MerkleTree, int]:
    S = 1 << log_s
    rs = ReedSolomon(message_len=S, blowup=blowup, dtype=F)
    _, _, tree = koalabear16_merkle()
    return BasefoldProver(rs, tree), rs, tree, S


def _columns(mle: jnp.ndarray) -> list[jnp.ndarray]:
    """The seam commits a Sequence of column MLEs (1D), like kzg/fri."""
    return [mle[:, j] for j in range(mle.shape[1])]


class BasefoldTest(absltest.TestCase):
    def test_commit_matches_independent_encode_merkle(self) -> None:
        bf, rs, tree, S = _basefold()
        K = 3
        mle = jnp.arange(S * K, dtype=F).reshape(S, K)  # [S, K]
        root, pdata = bf.commit(_columns(mle))
        # Independent reconstruction: column-wise RS-encode then Merkle.
        codeword = rs.encode(mle.T).T  # [S*blowup, K]
        exp_root, _ = tree.commit(codeword)
        self.assertEqual(root.tolist(), exp_root.tolist())
        self.assertIsInstance(pdata, BasefoldProverData)
        self.assertEqual(pdata.widths, (K,))

    def test_open_not_implemented(self) -> None:
        bf, *_ = _basefold()
        stub = StubTranscript(jnp.zeros(1, dtype=F))
        with self.assertRaises(NotImplementedError):
            bf.open(BasefoldProverData([], (0,)), [], stub)

    def test_prover_data_pytree_round_trips(self) -> None:
        bf, rs, tree, S = _basefold()
        mle = jnp.arange(S * 3, dtype=F).reshape(S, 3)
        _, pdata = bf.commit(_columns(mle))
        leaves, treedef = jax.tree_util.tree_flatten(pdata)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(restored.widths, pdata.widths)
        for a, b in zip(restored.digest_layers, pdata.digest_layers):
            self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_verify_not_implemented(self) -> None:
        _, rs, tree, _ = _basefold()
        stub = StubTranscript(jnp.zeros(1, dtype=F))
        with self.assertRaises(NotImplementedError):
            BasefoldVerifier(rs, tree).verify(
                jnp.zeros((), dtype=F), [], jnp.zeros(0, dtype=F), None, stub
            )


if __name__ == "__main__":
    absltest.main()
