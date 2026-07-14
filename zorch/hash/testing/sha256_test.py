# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 byte-hash — byte-match against the universal reference `hashlib.sha256`.

Agnostic golden: `hashlib` is the FIPS 180-4 reference, named by no consumer. The
lengths exercise every padding boundary — empty, sub-block, the 55/56 one-vs-two
block transition (where the 8-byte length field forces a second block), exact
block multiples, and a multi-block message.
"""
from __future__ import annotations

import functools
import hashlib

import frx
import frx.numpy as jnp
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
        marked = np.asarray(sha256.sha256_chain(sha256.INITIAL_STATE, blocks))
        state = jnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        inline = np.asarray(sha256.serialize_digest(sha256.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_emits_single_composite_marker(self) -> None:
        # digest lowers to exactly one stablehlo.composite, name-routed to the
        # dedicated zorch.sha256 emitter (parallel to zorch.poseidon2).
        blocks = jnp.asarray(sha256._pad(np.arange(64, dtype=np.uint8)[None, :]))
        fn = functools.partial(sha256.sha256_chain, sha256.INITIAL_STATE)
        txt = frx.jit(fn).lower(blocks).as_text()
        self.assertIn(sha256.SHA256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_serialize_deserialize_roundtrip(self) -> None:
        # deserialize_digest inverts serialize_digest, so unpacking a digest
        # recovers the exact midstate a stream resumes from.
        rng = np.random.default_rng(0)
        state = jnp.asarray(rng.integers(0, 2**32, (3, 8), np.int64).astype(np.uint32))
        back = sha256.deserialize_digest(sha256.serialize_digest(state))
        np.testing.assert_array_equal(np.asarray(back), np.asarray(state))

    @parameterized.parameters(1, 2, 3)
    def test_chain_resumes_from_midstate(self, split: int) -> None:
        # sha256_chain from a non-IV midstate resumes the compression: hashing a
        # 4-block message in two chained halves must equal one chain over all 4.
        blocks = jnp.asarray(
            np.random.default_rng(split)
            .integers(0, 2**32, (1, 4, 16), np.int64)
            .astype(np.uint32)
        )
        whole = sha256.sha256_chain(sha256.INITIAL_STATE, blocks)
        mid = sha256.deserialize_digest(
            sha256.sha256_chain(sha256.INITIAL_STATE, blocks[:, :split])
        )[0]
        resumed = sha256.sha256_chain(mid, blocks[:, split:])
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))

    def test_compress_explicit_k_matches_default(self) -> None:
        # Threading the round-constant table as an explicit `k` operand (what the
        # marked region does) matches the module-default `_Kd`.
        blocks = jnp.asarray(sha256._pad(np.arange(80, dtype=np.uint8)[None, :]))
        state = jnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        default = sha256.compress(state, blocks)
        explicit = sha256.compress(state, blocks, sha256._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


if __name__ == "__main__":
    absltest.main()
