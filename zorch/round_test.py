# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.round import Round
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


class RoundBaseTest(absltest.TestCase):
    def test_commit_delegates_to_observe(self):
        r = Round()
        t = StubTranscript(jnp.array([1, 2], dtype=KB))
        # observe is a no-op stub; commit must return a transcript unchanged in pos
        t2 = r.commit(t, jnp.array([9, 9], dtype=KB))
        self.assertEqual(t2.pos, 0)

    def test_challenge_delegates_to_sample(self):
        r = Round()
        t = StubTranscript(jnp.array([7, 8], dtype=KB))
        t2, c = r.challenge(t, 1)
        self.assertTrue(bool(c[0] == jnp.array(7, KB)))
        self.assertEqual(t2.pos, 1)

    def test_call_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Round()(None, None)


if __name__ == "__main__":
    absltest.main()
