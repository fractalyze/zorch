# zorch/commit/basefold_test.py
"""Basefold.commit — RS low-degree-extension + Merkle, against an independent
reconstruction of the same codeword + root."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.basefold import Basefold, BasefoldProverData
from zorch.commit.merkle import MerkleTree
from zorch.commit.pcs import Pcs
from zorch.commit.testing.koalabear16 import koalabear16_merkle


def _basefold(
    log_s: int = 2, blowup: int = 2
) -> tuple[Basefold, ReedSolomon, MerkleTree, int]:
    S = 1 << log_s
    rs = ReedSolomon(message_len=S, blowup=blowup, dtype=F)
    _, _, tree = koalabear16_merkle()
    return Basefold(rs, tree), rs, tree, S


class BasefoldTest(absltest.TestCase):
    def test_is_pcs(self) -> None:
        bf, *_ = _basefold()
        self.assertIsInstance(bf, Pcs)

    def test_commit_matches_independent_encode_merkle(self) -> None:
        bf, rs, tree, S = _basefold()
        K = 3
        mle = jnp.arange(S * K, dtype=F).reshape(S, K)  # [S, K]
        root, pdata = bf.commit(mle)
        # Independent reconstruction: column-wise RS-encode then Merkle.
        codeword = rs.encode(mle.T).T  # [S*blowup, K]
        exp_root, _ = tree.commit(codeword)
        self.assertEqual(root.tolist(), exp_root.tolist())
        self.assertIsInstance(pdata, BasefoldProverData)
        self.assertEqual(pdata.widths, (K,))

    def test_open_not_implemented(self) -> None:
        bf, *_ = _basefold()
        with self.assertRaises(NotImplementedError):
            bf.open(None, None, None)

    def test_prover_data_pytree_round_trips(self) -> None:
        bf, rs, tree, S = _basefold()
        mle = jnp.arange(S * 3, dtype=F).reshape(S, 3)
        _, pdata = bf.commit(mle)
        leaves, treedef = jax.tree_util.tree_flatten(pdata)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(restored.widths, pdata.widths)
        for a, b in zip(restored.digest_layers, pdata.digest_layers):
            self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_verify_not_implemented(self) -> None:
        bf, *_ = _basefold()
        with self.assertRaises(NotImplementedError):
            bf.verify(None, None, None, None, None)


if __name__ == "__main__":
    absltest.main()
