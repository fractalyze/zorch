# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


class StubTranscriptTest(absltest.TestCase):
    def test_sample_returns_preset_in_order(self):
        ch = jnp.array([10, 20, 30], dtype=KB)
        t = StubTranscript(ch)
        t, a = t.sample(1)
        t, b = t.sample(1)
        self.assertTrue(bool(a[0] == jnp.array(10, KB)))
        self.assertTrue(bool(b[0] == jnp.array(20, KB)))

    def test_observe_is_noop_and_immutable(self):
        ch = jnp.array([10, 20], dtype=KB)
        t0 = StubTranscript(ch)
        t1 = t0.observe(jnp.array([1, 2, 3], dtype=KB))
        # observe does not advance the challenge position
        _, a = t1.sample(1)
        self.assertTrue(bool(a[0] == jnp.array(10, KB)))
        # original transcript is unchanged (pos still 0)
        self.assertEqual(t0.pos, 0)


if __name__ == "__main__":
    absltest.main()
