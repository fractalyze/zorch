# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte pins for every hash zorch consumes — the gate an frx pin bump crosses.

Retiring a hash marker changes how a permutation lowers, not what it computes —
only the bytes tell the two apart, and a single byte of drift silently breaks
every verifier holding an older proof. Each pin below is the exact byte string a
consumed surface produces today, in canonical wire form rather than as field
values, since a representation change breaks a verifier exactly as a changed
value does. Captured against the frx pin in `requirements.in`
(0.10.0.dev20260720022913).

Pinned surfaces, one per hash zorch consumes:

* the algebraic transcript sponge — `DuplexTranscript` over Poseidon2
* the Merkle digest — `Sponge` leaves folded by `Compression`
* the byte transcript — `Sha256FieldTranscript`, plus `ByteHashTranscript` on
  BOTH substrates, so the device marker (`Sha256`, a retirement target) and
  plain `hashlib` (`HostSha256`, which no retirement can touch) are held to the
  same pin

Two things would keep this file off the pin-bump PR where a lowering change
actually arrives, so it carries neither: a `local_only` tag, which `.bazelrc.ci`
drops from CI, and a composite-`vmap`, for which CI's published frxlib has no
batching rule (see `commit/testing/merkle_test.py`). The Merkle pin therefore
folds the tree with unbatched `Sponge` / `Compression` calls rather than
`MerkleTree.commit`.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.byte_transcript import ByteHashTranscript
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sha256 import HostSha256, Sha256
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.sha256_field_transcript import Sha256FieldTranscript
from zorch.transcript import DuplexTranscript

F = zk_dtypes.koalabear_mont  # the koalabear16 permutation's field

# The domain and observed payload every pin below is captured over. Fixed and
# arbitrary: a pin needs a reproducible input, not a meaningful one.
_DOMAIN = b"zorch/hash-byte-pin"
_OBSERVED = np.arange(12, dtype=np.uint32)

# --- pins -------------------------------------------------------------------
# Poseidon2 duplex sponge: observe `_OBSERVED` as field elements, then two
# squeezes of 4 (the second pins the re-absorb of the first — a sponge that
# never duplexed would still match on the first squeeze alone).
_DUPLEX_CHALLENGES = (
    "b02386375abb4d39a5476f0a01111d2d",
    "921ed85fcd88b925296c6e101c3a844c",
)

# Sponge(rate=8, out=8) leaves over arange(32).reshape(4, 8), folded pairwise by
# Compression(arity=2, chunk=8) — byte-for-byte the Plonky3 root `merkle_test`
# pins `MerkleTree.commit` to, so the vmap is all that separates the two.
_MERKLE_ROOT = "06e194632d5f101ae70966010bcbfa2542fede61ee7169158eda6509b5c30b23"

# SHA-256 Merlin duplex: observe `_OBSERVED` slice-framed, squeeze 8 elements.
_SHA256_CHALLENGE = "2d93d9b73c925b20f4e6d302381e6566713bcc86e079306a22a45c97335f4f5e"


def _canonical_hex(values: Array) -> str:
    """Field elements as canonical little-endian u32 bytes. Canonical, not the
    in-memory Montgomery form: the wire carries the residue, so that is what a
    downstream verifier would see change."""
    return np.asarray(values.astype(fnp.uint32)).astype("<u4").tobytes().hex()


class TranscriptSpongeBytePinTest(absltest.TestCase):
    def test_duplex_challenges_match_the_pin(self) -> None:
        t = DuplexTranscript.new(koalabear16_perm(), rate=8)
        t = t.observe(fnp.asarray(_OBSERVED).astype(F))
        t, first = t.sample(4)
        _, second = t.sample(4)
        self.assertEqual(
            (_canonical_hex(first), _canonical_hex(second)), _DUPLEX_CHALLENGES
        )


class MerkleDigestBytePinTest(absltest.TestCase):
    def test_root_matches_the_pin(self) -> None:
        perm = koalabear16_perm()
        sponge = Sponge(perm, SpongeParams(rate=8, out=8))
        comp = Compression(perm, CompressionParams(arity=2, chunk=8))
        matrix = fnp.arange(32, dtype=F).reshape(4, 8)

        layer = [sponge.hash(matrix[i]) for i in range(matrix.shape[0])]
        while len(layer) > 1:
            layer = [
                comp.compress(fnp.stack(layer[i : i + 2]))
                for i in range(0, len(layer), 2)
            ]
        self.assertEqual(_canonical_hex(layer[0]), _MERKLE_ROOT)


class ByteTranscriptBytePinTest(absltest.TestCase):
    def test_field_surface_challenges_match_the_pin(self) -> None:
        t = Sha256FieldTranscript.new(_DOMAIN, np.uint32)
        _, c = t.observe(fnp.asarray(_OBSERVED)).sample(8)
        self.assertEqual(np.asarray(c).astype("<u4").tobytes().hex(), _SHA256_CHALLENGE)

    def test_both_substrates_match_the_pin(self) -> None:
        # The device marker and hashlib must agree with each other AND with the
        # pin: a retirement can only move the marker arm, so a drift there shows
        # up as one arm leaving the pin.
        payload = _OBSERVED.astype("<u4").tobytes()
        for substrate in (HostSha256(), Sha256()):
            with self.subTest(substrate=type(substrate).__name__):
                t = ByteHashTranscript.new(_DOMAIN, substrate).observe_slice(
                    payload, _OBSERVED.size
                )
                _, squeezed = t.sample_slice(8, 4)
                self.assertEqual(squeezed.hex(), _SHA256_CHALLENGE)


if __name__ == "__main__":
    absltest.main()
