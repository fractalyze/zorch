# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.round import Round
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


class RoundBaseTest(absltest.TestCase):
    def test_observe_delegates_to_transcript(self):
        # StubTranscript.observe is a no-op, so a mock is needed to assert delegation.
        r = Round()
        t = MagicMock()
        values = jnp.array([9, 9], dtype=KB)
        t2 = r.observe(t, values)
        t.observe.assert_called_once_with(values)
        self.assertIs(t2, t.observe.return_value)

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
