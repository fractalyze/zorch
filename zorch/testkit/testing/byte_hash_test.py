# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The host oracles are the differential partner, so they are pinned externally.

Checking `HostSha256` against `hashlib` would restate its own body. What is worth
pinning is what the `ByteHash` seam asks of an implementation and the transcript
suites do not reach: published standard vectors (the oracle is only an oracle if
it is right about the standard, not about our wrapper), the per-message batch
contract, the value semantics a row carried as pytree aux needs, and — for the
XOF — a width other than the default 32 the suites always use.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from zorch.testkit.byte_hash import HostBlake3, HostSha256

# FIPS 180-4 B.1: SHA-256 of b"abc".
_SHA256_ABC = bytes.fromhex(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
)
# BLAKE3 official test vector, `input_len` 3 — the project's inputs are
# `input[i] = i % 251`, so this message is bytes 0x00..0x02. Quoted to 64 bytes
# so the XOF read below is pinned against the standard, not against itself.
_BLAKE3_3_XOF64 = bytes.fromhex(
    "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"
    "5b49b82f805a538c68915c1ae8035c900fd1d4b13902920fd05e1450822f36de"
)
_BLAKE3_MSG = bytes(range(3))


def _rows(*msgs: bytes) -> np.ndarray:
    """Equal-length messages as the uint8 `[B, L]` the seam takes."""
    return np.array([list(m) for m in msgs], dtype=np.uint8)


class HostSha256Test(absltest.TestCase):
    def test_matches_the_fips_vector(self) -> None:
        digest = HostSha256().digest(_rows(b"abc"))
        self.assertEqual(bytes(digest[0]), _SHA256_ABC)

    def test_digest_size_is_the_sha256_width(self) -> None:
        self.assertEqual(HostSha256().digest_size, 32)

    def test_hashes_each_message_of_a_batch_independently(self) -> None:
        batch = HostSha256().digest(_rows(b"abc", b"xyz"))
        self.assertEqual(batch.shape, (2, 32))
        self.assertEqual(batch.dtype, np.uint8)
        self.assertEqual(bytes(batch[0]), _SHA256_ABC)
        # Row 1 is its own hash, not a chain over row 0.
        self.assertEqual(bytes(batch[1]), bytes(HostSha256().digest(_rows(b"xyz"))[0]))

    def test_compares_by_value(self) -> None:
        # The seam's rule for every implementation: a row carried as pytree aux
        # that compared by identity would re-trace its enclosing zone per
        # freshly built instance.
        self.assertEqual(HostSha256(), HostSha256())
        self.assertEqual(hash(HostSha256()), hash(HostSha256()))

    def test_is_not_equal_to_another_row(self) -> None:
        self.assertNotEqual(HostSha256(), HostBlake3())


class HostBlake3Test(absltest.TestCase):
    def test_matches_the_reference_vector(self) -> None:
        digest = HostBlake3().digest(_rows(_BLAKE3_MSG))
        self.assertEqual(bytes(digest[0]), _BLAKE3_3_XOF64[:32])

    def test_reads_the_requested_width(self) -> None:
        # BLAKE3 is an XOF, so the width is free — and the standard's own
        # extended output says what each width must contain. The transcript
        # suites only ever read the default 32.
        row = _rows(_BLAKE3_MSG)
        for width in (1, 16, 64):
            digest = HostBlake3(width).digest(row)
            self.assertEqual(digest.shape, (1, width))
            self.assertEqual(bytes(digest[0]), _BLAKE3_3_XOF64[:width])

    def test_rejects_a_non_positive_width(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                HostBlake3(bad)

    def test_compares_on_its_width(self) -> None:
        self.assertEqual(HostBlake3(16), HostBlake3(16))
        self.assertEqual(hash(HostBlake3(16)), hash(HostBlake3(16)))
        self.assertNotEqual(HostBlake3(16), HostBlake3(32))


if __name__ == "__main__":
    absltest.main()
