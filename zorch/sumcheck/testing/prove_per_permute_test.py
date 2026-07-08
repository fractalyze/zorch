# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.hash.poseidon2.poseidon2 import POSEIDON2_MARKER
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.sumcheck import prover
from zorch.sumcheck.prover import prove
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont


class ProveTest(absltest.TestCase):
    def test_rejects_zero_round_state(self) -> None:
        # A width-1 carry derives 0 rounds: the scan would yield no round polys.
        # Fail fast with a clear message instead.
        with self.assertRaisesRegex(ValueError, "at least one round"):
            prove(
                prover.SumcheckRound(1), [jnp.arange(1, dtype=KB)], cheap_transcript(KB)
            )

    def test_stacks_sumcheck_messages(self) -> None:
        f = jnp.arange(1, 17, dtype=KB)
        _, _, msgs = prove(prover.SumcheckRound(degree=1), [f], cheap_transcript(KB))
        self.assertEqual(msgs.round_poly.shape, (4, 2))  # n rounds × (degree+1)
        self.assertEqual(msgs.challenge.shape, (4,))  # one challenge per round


class ProvePerPermuteTest(absltest.TestCase):
    """`prove` always runs the plain per-variable scan. The whole-scan
    `zorch.sumcheck` megakernel that once wrapped the entire scan register-resident
    was dropped (it ptxas-overflowed shared memory around 2^20, so it never compiled
    at real sizes). Even over a transcript whose permutation has a dedicated fusion
    marker (real poseidon2), Fiat-Shamir stays a per-permute `zorch.poseidon2` marker
    — one fused kernel per permute — with no whole-scan composite wrapping it.

    Plain-scan correctness (product / extension challenges / `eval_start`) is
    covered against the reference oracles in `prove_test.py`; these cases guard only
    that the megakernel stays gone and the per-permute FS marker survives."""

    def _poseidon_transcript(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def test_per_permute_fs_marker_over_poseidon_transcript(self) -> None:
        # Over a real poseidon2 transcript the lowering carries the per-permute
        # `zorch.poseidon2` FS marker (that no whole-scan composite wraps the
        # scan is guarded structurally by test_no_top_level_composite below).
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        text = jax.jit(lambda x, y: prove(rnd, [x, y], t0)).lower(a, b).as_text()
        self.assertIn(f'"{POSEIDON2_MARKER}"', text)

    def test_no_top_level_composite(self) -> None:
        # The driver lowers to a `lax.scan` (the per-variable loop), not a top-level
        # composite: the per-permute poseidon2 markers ride inside the scan body, so
        # no composite wraps the whole scan — a regression guard the wrapper is gone.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(
            lambda x, y: prove(prover.SumcheckRound(degree=2), [x, y], t0)
        )(a, b).jaxpr
        self.assertFalse(any(e.primitive.name == "composite" for e in jaxpr.eqns))


if __name__ == "__main__":
    absltest.main()
