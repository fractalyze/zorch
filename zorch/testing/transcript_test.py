# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, tree_util

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript, StubTranscript

KB = zk_dtypes.koalabear
F = zk_dtypes.koalabear_mont  # the koalabear16 permutation's field

# The duplex sponge's `lax.scan` absorb is correct on GPU but hits a ZKX CPU
# while-emitter bug that drops the scan's array-carry update (eager runs go
# non-deterministic). Skip the duplex tests on CPU until it lands; the stub
# transcript tests above stay on. See fractalyze/zkx#500.
_CPU_BACKEND = jax.default_backend() == "cpu"


class StubTranscriptTest(absltest.TestCase):
    def test_sample_returns_preset_in_order(self) -> None:
        ch = jnp.array([10, 20, 30], dtype=KB)
        t = StubTranscript(ch)
        t, a = t.sample(1)
        t, b = t.sample(1)
        self.assertTrue(bool(a[0] == jnp.array(10, KB)))
        self.assertTrue(bool(b[0] == jnp.array(20, KB)))

    def test_observe_is_noop_and_immutable(self) -> None:
        ch = jnp.array([10, 20], dtype=KB)
        t0 = StubTranscript(ch)
        t1 = t0.observe(jnp.array([1, 2, 3], dtype=KB))
        # observe does not advance the challenge position
        _, a = t1.sample(1)
        self.assertTrue(bool(a[0] == jnp.array(10, KB)))
        # original transcript is unchanged (pos still 0)
        self.assertEqual(t0.pos, 0)

    def test_observe_and_sample_matches_observe_then_sample(self) -> None:
        ch = jnp.array([10, 20, 30], dtype=KB)
        v = jnp.array([1, 2, 3], dtype=KB)
        t_ref, ref = StubTranscript(ch).observe(v).sample(2)
        t_fused, fused = StubTranscript(ch).observe_and_sample(v, 2)
        self.assertTrue(bool(jnp.all(ref == fused)))
        self.assertEqual(t_ref.pos, t_fused.pos)

    def test_is_pytree(self) -> None:
        # challenges + pos are the two leaves (pos a leaf, not static, so the
        # transcript survives as a lax.scan carry); flatten/unflatten round-trips.
        t = StubTranscript(jnp.array([10, 20, 30], dtype=KB))
        leaves, treedef = tree_util.tree_flatten(t)
        self.assertEqual(len(leaves), 2)
        _, a = tree_util.tree_unflatten(treedef, leaves).sample(1)
        self.assertTrue(bool(a[0] == jnp.array(10, KB)))

    def test_threads_through_jit_as_a_scan_carry(self) -> None:
        # The capability registration buys: sample under a lax.scan that carries
        # the stub, advancing pos each step. Backend-agnostic (no field scatter).
        ch = jnp.array([10, 20, 30, 40], dtype=KB)

        def step(t: StubTranscript, _: Array) -> tuple[StubTranscript, Array]:
            t, x = t.sample(1)
            return t, x[0]

        _, got = jax.lax.scan(step, StubTranscript(ch), xs=None, length=4)
        self.assertTrue(bool(jnp.all(got == ch)))


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


if __name__ == "__main__":
    absltest.main()
