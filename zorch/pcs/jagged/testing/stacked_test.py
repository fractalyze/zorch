# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Oracle: stacked_open eval == eval_mle(d_full, z_final); stacked_verify passes."""
from __future__ import annotations

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.prover import BasefoldProver
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.pcs.jagged.dense import JaggedLayout
from zorch.pcs.jagged.stacked import stacked_open, stacked_verify
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return rand_field(seed, (*shape, 4), F).view(EF).reshape(shape)


def _t() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


class StackedTest(absltest.TestCase):
    def test_stacked_open_matches_full_mle_eval(self) -> None:
        log_s, log_k = 3, 2
        S, K = 1 << log_s, 1 << log_k
        rs = ReedSolomon(message_len=S, blowup=2, dtype=EF)
        _, _, tree = koalabear16_merkle()
        prover = BasefoldProver(rs, tree, num_queries=4)
        verifier = BasefoldVerifier(rs, tree, num_queries=4)
        mle = _rand_ef(1, (S, K))
        root, pdata = prover.commit([mle[:, k] for k in range(K)])
        layout = JaggedLayout(
            log_m=log_s + log_k,
            log_s=log_s,
            heights=(S,) * K,
            widths=(1,) * K,
            dtype=EF,
        )
        z_final = _rand_ef(2, (log_s + log_k,))
        dense_eval, values, proof, _ = stacked_open(
            prover, pdata, z_final, layout, _t()
        )
        # d_full mirrors commit.py's reshape(K, S).T: mle is [S,K],
        # so mle.T is [K,S], and mle.T.reshape(-1) lays out k*S+s order (K-bits MSB).
        d_full = mle.T.reshape(-1)
        self.assertEqual(dense_eval.tolist(), eval_mle(d_full, z_final).tolist())
        ok, _ = stacked_verify(
            verifier, root, z_final, dense_eval, values, proof, layout, _t()
        )
        self.assertTrue(bool(ok))


if __name__ == "__main__":
    absltest.main()
