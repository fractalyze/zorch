# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 byte-hash — byte-match against the universal reference `hashlib.sha256`.

Agnostic golden: `hashlib` is the FIPS 180-4 reference, named by no consumer. The
lengths exercise every padding boundary — empty, sub-block, the 55/56 one-vs-two
block transition (where the 8-byte length field forces a second block), exact
block multiples, and a multi-block message.
"""
from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from zorch.hash import sha256

# Padding-boundary lengths: 0/1 (empty + tiny), 55/56 (the one-block/two-block
# cutoff), 63/64 (block edge), 119/120 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 119, 120)


class Sha256Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha256.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha256(bytes(msg)).digest())

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must equal
        # the per-message hashlib digests, in order.
        length = 64
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, length), dtype=np.uint8)
        got = np.asarray(sha256.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha256(bytes(batch[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_marked_equals_inline(self, length: int) -> None:
        # The zorch.sha256 marker only tags the region; with no dedicated emitter
        # wired it inlines its decomposition, so the marked digest must byte-equal
        # the unmarked compression at every padding boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = jnp.asarray(sha256._pad(msg[None, :]))
        marked = np.asarray(sha256._digest_words_marked(blocks))
        inline = np.asarray(sha256._digest_words(blocks))
        np.testing.assert_array_equal(marked, inline)

    def test_emits_single_composite_marker(self) -> None:
        # digest lowers to exactly one stablehlo.composite, name-routed to the
        # dedicated zorch.sha256 emitter (parallel to zorch.poseidon2).
        blocks = jnp.asarray(sha256._pad(np.arange(64, dtype=np.uint8)[None, :]))
        txt = jax.jit(sha256._digest_words_marked).lower(blocks).as_text()
        self.assertIn(sha256.SHA256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)


if __name__ == "__main__":
    absltest.main()
