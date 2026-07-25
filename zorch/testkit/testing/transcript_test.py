# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest

from zorch.hash.permutation import Permutation
from zorch.testkit.transcript import CheapPermutation, cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont


class CheapPermutationTest(absltest.TestCase):
    def test_satisfies_permutation_protocol(self) -> None:
        # runtime_checkable: width / dtype / has_dedicated_fusion / permute.
        self.assertIsInstance(CheapPermutation(width=8, dtype=KB), Permutation)

    def test_stays_off_the_dedicated_fusion_path(self) -> None:
        # The whole point: prove gates marking on this, so a test transcript must
        # report False and stay on the unmarked path.
        self.assertFalse(CheapPermutation(width=8, dtype=KB).has_dedicated_fusion)

    def test_permute_preserves_shape_and_is_deterministic(self) -> None:
        perm = CheapPermutation(width=8, dtype=KB)
        s = fnp.arange(8, dtype=KB)
        out, again = perm.permute(s), perm.permute(s)
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, KB)
        self.assertTrue(bool(fnp.all(out == again)))

    def test_value_equality(self) -> None:
        # Pytree-aux seat: fresh instances must compare by value or every jit
        # zone taking a cheap transcript re-traces per call.
        self.assertEqual(
            CheapPermutation(width=8, dtype=KB), CheapPermutation(width=8, dtype=KB)
        )
        self.assertEqual(
            hash(CheapPermutation(width=8, dtype=KB)),
            hash(CheapPermutation(width=8, dtype=KB)),
        )
        self.assertNotEqual(
            CheapPermutation(width=8, dtype=KB), CheapPermutation(width=4, dtype=KB)
        )


class TestTranscriptTest(absltest.TestCase):
    def test_returns_duplex_transcript(self) -> None:
        self.assertIsInstance(cheap_transcript(KB), DuplexTranscript)

    def test_is_deterministic_across_fresh_instances(self) -> None:
        # Same observe/sample sequence on two fresh transcripts → same challenge,
        # so a test can drive prove and a reference from identical fresh state.
        _, a = cheap_transcript(KB).observe_and_sample(fnp.arange(3, dtype=KB), 1)
        _, b = cheap_transcript(KB).observe_and_sample(fnp.arange(3, dtype=KB), 1)
        self.assertTrue(bool(fnp.all(a == b)))

    def test_observation_changes_the_challenge(self) -> None:
        # A real sponge (observe is not a no-op): a different observation must
        # yield a different challenge.
        _, a = cheap_transcript(KB).observe_and_sample(fnp.arange(3, dtype=KB), 1)
        _, b = cheap_transcript(KB).observe_and_sample(fnp.arange(3, 6, dtype=KB), 1)
        self.assertFalse(bool(fnp.all(a == b)))


if __name__ == "__main__":
    absltest.main()
