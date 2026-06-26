# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`DuplexTranscript.fs_on_host` routes the duplex sponge to the host CPU, keeping
the sponge state host-resident across the stream -- byte-identical to the on-device
sponge. Every test compares `fs_on_host=True` against the default device path on the
same inputs."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import tree_util

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript, sample_challenge

F = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont

# Host-FS runs the sponge on the CPU inside the callback; on a CPU-only backend the
# device and host sponges coincide (nothing to compare) and the run also hits the
# ZKX CPU scan array-carry bug the duplex tests skip for (fractalyze/zkx#500).
_CPU_BACKEND = jax.default_backend() == "cpu"


@absltest.skipIf(
    _CPU_BACKEND, "host-FS vs device sponge is only meaningful off CPU (zkx#500)"
)
class TranscriptHostFsTest(parameterized.TestCase):
    """The `fs_on_host` opt-in: observe/sample/observe_and_sample/sample_challenge
    route to the host sponge and stay byte-identical to the device path."""

    def _new(self, fs_on_host: bool) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8, fs_on_host=fs_on_host)

    def _state_eq(self, a: DuplexTranscript, b: DuplexTranscript) -> bool:
        return all(
            bool(jnp.all(x == y))
            for x, y in zip(tree_util.tree_leaves(a), tree_util.tree_leaves(b))
        )

    @parameterized.parameters(3, 8, 19)  # partial block, full block, block-crossing
    def test_observe_byte_identical(self, n: int) -> None:
        v = rand_field(1, (n,), F)
        self.assertTrue(
            self._state_eq(self._new(False).observe(v), self._new(True).observe(v))
        )

    @parameterized.parameters(1, 4, 9)
    def test_sample_byte_identical(self, k: int) -> None:
        v = rand_field(2, (5,), F)
        ta, dev = self._new(False).observe(v).sample(k)
        tb, host = self._new(True).observe(v).sample(k)
        self.assertTrue(bool(jnp.all(dev == host)))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.parameters(1, 4)
    def test_observe_and_sample_byte_identical(self, k: int) -> None:
        # The single-callback fused absorb+squeeze == the device fused form.
        v = rand_field(6, (5,), F)
        ta, dev = self._new(False).observe_and_sample(v, k)
        tb, host = self._new(True).observe_and_sample(v, k)
        self.assertTrue(bool(jnp.all(dev == host)))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.parameters(0, 3)
    def test_check_witness_byte_identical(self, pow_bits: int) -> None:
        # check_witness routes through the backend too -- it once ran the device
        # sponge regardless of fs_on_host, so a host-FS grind's re-check could
        # disagree with its own search. Host and device must match.
        w = rand_field(8, (1,), F)[0]
        ta, dev = self._new(False).check_witness(pow_bits, w)
        tb, host = self._new(True).check_witness(pow_bits, w)
        self.assertEqual(bool(dev), bool(host))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.named_parameters(("base_1limb", F, 1), ("ext_4limb", EF, 4))
    def test_sample_challenge_byte_identical(self, dtype: Any, limbs: int) -> None:
        # sample_challenge routes through the `sample` method, so it picks up the
        # host body; the multi-limb reinterpret stays on the device.
        v = rand_field(3, (5,), F)
        _, dev = sample_challenge(self._new(False).observe(v), dtype, limbs)
        _, host = sample_challenge(self._new(True).observe(v), dtype, limbs)
        self.assertTrue(bool(jnp.all(dev == host)))

    def test_flag_carries_across_steps(self) -> None:
        # fs_on_host must survive every step so the whole stream stays on host.
        t = self._new(True)
        self.assertTrue(t.observe(rand_field(4, (4,), F)).fs_on_host)
        self.assertTrue(t.sample(1)[0].fs_on_host)
        self.assertTrue(t.observe_and_sample(rand_field(5, (3,), F), 2)[0].fs_on_host)

    def test_state_goes_host_resident(self) -> None:
        # The optimization: the host-FS sponge state lives on the CPU across the
        # whole stream, so each hop crosses only `values` in and the challenge out,
        # not the 5 state leaves both ways.
        t = self._new(True).observe(rand_field(7, (4,), F))
        leaf = tree_util.tree_leaves(t)[0]
        self.assertEqual(next(iter(leaf.devices())).platform, "cpu")


if __name__ == "__main__":
    absltest.main()
