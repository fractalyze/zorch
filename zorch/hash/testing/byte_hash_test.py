# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The `ByteHash` seam and its SHA-256 implementations.

Both `Sha256` (device marker) and `HostSha256` (host) must produce the identical
FIPS 180-4 bytes — pinned against the universal reference `hashlib.sha256`, named
by no consumer — and satisfy the `ByteHash` protocol with stable value identity
(so the seam is re-trace-safe as pytree aux).
"""
from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest, parameterized

from zorch.hash.byte_hash import ByteHash
from zorch.hash.sha256 import HostSha256, Sha256

# Padding-boundary lengths: 0/1, the 55/56 one-vs-two-block cutoff, 63/64 block
# edge, 119/120 multi-block.
_LENGTHS = (1, 55, 56, 63, 64, 119, 120)
_IMPLS = (Sha256(), HostSha256())


class ByteHashTest(parameterized.TestCase):
    @parameterized.parameters(*_IMPLS)
    def test_is_byte_hash(self, h: ByteHash) -> None:
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 32)
        self.assertIsInstance(h.has_dedicated_fusion, bool)

    def test_fusion_flags(self) -> None:
        # The seam's whole point: the substrate axis lives on the hash, not a
        # class name. Marker lowers to a kernel; hashlib is host-eager.
        self.assertTrue(Sha256().has_dedicated_fusion)
        self.assertFalse(HostSha256().has_dedicated_fusion)

    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        rng = np.random.default_rng(length)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        ref = np.stack(
            [
                np.frombuffer(hashlib.sha256(r.tobytes()).digest(), np.uint8)
                for r in msgs
            ]
        )
        for h in _IMPLS:
            got = np.asarray(h.digest(msgs))
            self.assertEqual(got.shape, (4, 32))
            np.testing.assert_array_equal(
                got, ref, err_msg=f"{type(h).__name__} L={length}"
            )

    def test_marker_equals_hashlib_impl(self) -> None:
        msgs = np.random.default_rng(7).integers(0, 256, size=(8, 48), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(Sha256().digest(msgs)), np.asarray(HostSha256().digest(msgs))
        )

    def test_value_identity(self) -> None:
        # Param-free -> all instances of a type are equal (stable pytree aux); the
        # two hashes are never equal to each other.
        for cls in (Sha256, HostSha256):
            self.assertEqual(cls(), cls())
            self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(Sha256(), HostSha256())


if __name__ == "__main__":
    absltest.main()
