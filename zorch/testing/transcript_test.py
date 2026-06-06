# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import tree_util

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, sample_challenge

F = zk_dtypes.koalabear_mont  # the koalabear16 permutation's field

# The duplex sponge's `lax.scan` absorb is correct on GPU but hits a ZKX CPU
# while-emitter bug that drops the scan's array-carry update (eager runs go
# non-deterministic). Skip the duplex tests on CPU until it lands. See
# fractalyze/zkx#500.
_CPU_BACKEND = jax.default_backend() == "cpu"


@absltest.skipIf(
    _CPU_BACKEND,
    "ZKX CPU scan array-carry bug (GPU-correct); remove when fractalyze/zkx#500 lands",
)
class DuplexTranscriptTest(absltest.TestCase):
    """The real duplex-sponge transcript: a device-side JAX pytree that threads
    functionally under @jit. observe absorbs into the sponge; sample squeezes
    field elements derived from everything observed so far (Fiat-Shamir)."""

    def _new(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def test_sample_returns_n_field_elements(self) -> None:
        _, out = self._new().sample(3)
        self.assertEqual(out.shape, (3,))
        self.assertEqual(out.dtype, F)

    def test_deterministic_for_equal_observations(self) -> None:
        v = rand_field(1, (5,), F)
        _, a = self._new().observe(v).sample(2)
        _, b = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(a == b)))

    def test_challenge_binds_to_observation(self) -> None:
        # Fiat-Shamir: a changed transcript must yield different challenges.
        v = rand_field(2, (5,), F)
        _, ca = self._new().observe(v).sample(2)
        _, cb = self._new().observe(v.at[2].add(jnp.array(1, F))).sample(2)
        self.assertFalse(bool(jnp.all(ca == cb)))

    def test_sample_stream_is_consistent(self) -> None:
        # sample(2) equals two sample(1)s in sequence.
        t0 = self._new().observe(rand_field(3, (4,), F))
        _, both = t0.sample(2)
        t1, x0 = t0.sample(1)
        _, x1 = t1.sample(1)
        self.assertTrue(bool(both[0] == x0[0]))
        self.assertTrue(bool(both[1] == x1[0]))

    def test_threads_under_jit(self) -> None:
        # Acceptance: state threads functionally under @jit (so the transcript
        # can later live in a lax.scan carry, issue #58).
        v = rand_field(4, (5,), F)
        got = jax.jit(lambda t, x: t.observe(x).sample(2)[1])(self._new(), v)
        _, want = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_observe_and_sample_matches_observe_then_sample(self) -> None:
        # The fused per-round primitive is a drop-in for observe-then-sample:
        # identical challenges and identical resulting transcript state.
        v = rand_field(7, (5,), F)
        t_ref, ref = self._new().observe(v).sample(2)
        t_fused, fused = self._new().observe_and_sample(v, 2)
        self.assertTrue(bool(jnp.all(ref == fused)))
        for a, b in zip(tree_util.tree_leaves(t_ref), tree_util.tree_leaves(t_fused)):
            self.assertTrue(bool(jnp.all(a == b)))

    def test_observe_and_sample_fuses_under_one_jit(self) -> None:
        # Acceptance: absorb+squeeze are one @jit computation (fused by
        # construction), matching the eager observe-then-sample reference.
        v = rand_field(8, (5,), F)
        got = jax.jit(lambda t, x: t.observe_and_sample(x, 2)[1])(self._new(), v)
        _, want = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_is_pytree(self) -> None:
        # 5 state buffers are the leaves; permutation + rate are static.
        leaves, treedef = tree_util.tree_flatten(self._new())
        self.assertEqual(len(leaves), 5)
        rebuilt = tree_util.tree_unflatten(treedef, leaves)
        _, a = self._new().sample(1)
        _, b = rebuilt.sample(1)
        self.assertTrue(bool(a[0] == b[0]))


class SampleChallengeTest(absltest.TestCase):
    """`sample_challenge` over the cheap test sponge -- the squeeze rule
    itself, independent of the duplex permutation."""

    def test_single_limb_is_plain_sample(self) -> None:
        t = cheap_transcript(F)
        t_ch, ch = sample_challenge(t, F, 1)
        t_raw, raw = t.sample(1)
        self.assertTrue(bool(ch == raw[0]))
        # Both advance the sponge identically.
        _, a = t_ch.sample(1)
        _, b = t_raw.sample(1)
        self.assertTrue(bool(a[0] == b[0]))

    def test_multi_limb_reinterprets_base_samples_as_coefficients(self) -> None:
        EF = zk_dtypes.koalabearx4_mont
        t = cheap_transcript(F)
        _, ch = sample_challenge(t, EF, 4)
        _, raw = t.sample(4)
        self.assertEqual(ch.dtype, EF)
        self.assertTrue(bool(jnp.all(ch[None].view(F) == raw)))

    def test_rejects_mismatched_packing(self) -> None:
        # The squeezes are consumed before the reinterpret, so a wrong limb
        # count must fail loud rather than truncate to the first element.
        EF = zk_dtypes.koalabearx4_mont
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), EF, 8)
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), F, 2)
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), F, 0)


class TranscriptJitCacheTest(absltest.TestCase):
    """Fresh transcripts must be cache-key-equal: the permutation is pytree aux
    (meta_fields), so two `DuplexTranscript.new(...)` over independently built,
    value-equal permutations must yield IDENTICAL treedefs — otherwise every jit
    zone taking a transcript re-traces per call (issue #163: ~2 min/call on the
    jagged verify replay whose kernels run in 20 ms)."""

    def test_fresh_cheap_transcripts_share_treedef(self) -> None:
        # The cheap permutation has no jit-cache counterpart below (its scan
        # hits the zkx#500 CPU bug), so treedef equality is its only guard.
        self.assertEqual(
            tree_util.tree_structure(cheap_transcript(F)),
            tree_util.tree_structure(cheap_transcript(F)),
        )

    def test_jit_zone_does_not_retrace_per_fresh_transcript(self) -> None:
        @jax.jit
        def zone(t: DuplexTranscript) -> jnp.ndarray:
            return t.state.sponge_state

        zone(DuplexTranscript.new(koalabear16_perm(), rate=8))
        zone(DuplexTranscript.new(koalabear16_perm(), rate=8))
        # _cache_size() is a private JAX API; may change on jax upgrade.
        self.assertEqual(zone._cache_size(), 1)


if __name__ == "__main__":
    absltest.main()
