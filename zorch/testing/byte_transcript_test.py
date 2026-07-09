# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-duplex Fiat-Shamir transcript — contract + a self-anchored golden.

Agnostic: no consumer named, no field type. The structural tests pin the
domain-separation / non-collision properties; the golden is a ~20-line
independent re-implementation of the same Merlin-over-SHA-256 construction on
`hashlib`, so a drift in the wire framing is caught without an external fixture.
"""
from __future__ import annotations

import hashlib

from absl.testing import absltest

from zorch.byte_transcript import (
    KIND_SCALAR,
    KIND_SLICE,
    OP_BYTES,
    OP_DOMAIN,
    OP_LABEL,
    OP_OBSERVE,
    OP_SQUEEZE,
    ByteHashTranscript,
)
from zorch.hash.byte_hash import ByteHash
from zorch.hash.sha256 import HostSha256, Sha256


def _new(domain: bytes) -> ByteHashTranscript:
    """A byte transcript on the host `hashlib` substrate — the framing under test
    is hash-agnostic; `test_device_substrate_matches_host` pins the marker."""
    return ByteHashTranscript.new(domain, HostSha256())


def _le8(n: int) -> bytes:
    return int(n).to_bytes(8, "little")


# ---- independent reference: the same construction, written directly on bytes ----
def _ref_new(domain: bytes) -> bytes:
    return bytes([OP_DOMAIN]) + _le8(len(domain)) + domain


def _ref_squeeze(buf: bytes, n: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(buf + _le8(ctr)).digest()
        ctr += 1
    return out[:n]


def _ref_sample_scalar(buf: bytes, n: int) -> tuple[bytes, bytes]:
    buf = buf + bytes([OP_SQUEEZE, KIND_SCALAR])
    sq = _ref_squeeze(buf, n)
    return buf + sq, sq


def _ref_sample_slice(buf: bytes, count: int, width: int) -> tuple[bytes, bytes]:
    buf = buf + bytes([OP_SQUEEZE, KIND_SLICE]) + _le8(count)
    sq = _ref_squeeze(buf, count * width)
    return buf + sq, sq


class ByteTranscriptTest(absltest.TestCase):
    # ---- self-anchored golden against an independent hashlib reference ----
    def test_matches_independent_reference(self) -> None:
        t = _new(b"agnostic-domain")
        ref = _ref_new(b"agnostic-domain")

        t = t.observe_label(b"phase-A")
        ref += bytes([OP_LABEL]) + _le8(7) + b"phase-A"
        t = t.observe_bytes(b"\x00\x11\x22\x33")
        ref += bytes([OP_BYTES]) + _le8(4) + b"\x00\x11\x22\x33"
        t = t.observe_scalar(b"sixteen--bytes!!")
        ref += bytes([OP_OBSERVE, KIND_SCALAR]) + b"sixteen--bytes!!"
        t = t.observe_slice(b"AABBBBCCCCCCDDDD", 4)
        ref += bytes([OP_OBSERVE, KIND_SLICE]) + _le8(4) + b"AABBBBCCCCCCDDDD"

        t, s0 = t.sample_scalar(16)
        ref, r0 = _ref_sample_scalar(ref, 16)
        self.assertEqual(s0, r0)

        t, s1 = t.sample_slice(3, 16)  # spans 2 SHA-256 output blocks (48 > 32)
        ref, r1 = _ref_sample_slice(ref, 3, 16)
        self.assertEqual(s1, r1)
        self.assertEqual(t.buffer, ref)

    # ---- domain separation / divergence ----
    def test_different_domains_diverge(self) -> None:
        a = _new(b"dom-a").sample_scalar(16)[1]
        b = _new(b"dom-b").sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    def test_label_changes_output(self) -> None:
        base = _new(b"d")
        a = base.observe_label(b"L").sample_scalar(16)[1]
        b = base.sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    def test_different_observations_diverge(self) -> None:
        base = _new(b"d")
        a = base.observe_scalar(b"A" * 16).sample_scalar(16)[1]
        b = base.observe_scalar(b"B" * 16).sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    # ---- non-collision (the op/kind tags + length prefixes do their job) ----
    def test_scalar_vs_slice_of_one_dont_collide(self) -> None:
        v = b"X" * 16
        base = _new(b"d")
        a = base.observe_scalar(v).sample_scalar(16)[1]
        b = base.observe_slice(v, 1).sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    def test_two_scalars_vs_one_slice_of_two_dont_collide(self) -> None:
        x, y = b"1" * 16, b"2" * 16
        base = _new(b"d")
        a = base.observe_scalar(x).observe_scalar(y).sample_scalar(16)[1]
        b = base.observe_slice(x + y, 2).sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    def test_sample_scalar_vs_slice_of_one_differ(self) -> None:
        base = _new(b"d")
        a = base.sample_scalar(16)[1]
        b = base.sample_slice(1, 16)[1]
        self.assertNotEqual(a, b)

    def test_sample_advances_state(self) -> None:
        # A sample then an observe must diverge from the no-sample path — pins the
        # re-absorb of squeezed bytes.
        base = _new(b"d")
        a = base.sample_scalar(16)[0].observe_scalar(b"Z" * 16).sample_scalar(16)[1]
        b = base.observe_scalar(b"Z" * 16).sample_scalar(16)[1]
        self.assertNotEqual(a, b)

    # ---- proof-of-work ----
    def test_pow_roundtrip(self) -> None:
        for bits in (0, 5, 10, 14):
            p = _new(b"pow").observe_bytes(b"root")
            p, nonce = p.grind_pow(bits)
            v = _new(b"pow").observe_bytes(b"root")
            v, ok = v.verify_pow(nonce, bits)
            self.assertTrue(ok, f"verify failed at bits={bits}")
            # Subsequent challenges agree on both sides.
            self.assertEqual(p.sample_scalar(16)[1], v.sample_scalar(16)[1])

    def test_pow_rejects_wrong_nonce(self) -> None:
        p = _new(b"pow").observe_bytes(b"root")
        _, nonce = p.grind_pow(10)
        v = _new(b"pow").observe_bytes(b"root")
        _, ok = v.verify_pow(nonce + 1, 10)
        self.assertFalse(ok)

    def test_pow_zero_bits_requires_canonical_nonce(self) -> None:
        mk = lambda: _new(b"pow").observe_bytes(b"root")
        self.assertEqual(mk().grind_pow(0)[1], 0)
        self.assertTrue(mk().verify_pow(0, 0)[1])
        for bad in (1, 42, 2**64 - 1):
            self.assertFalse(mk().verify_pow(bad, 0)[1])

    # ---- the marker (device) substrate is byte-identical to host hashlib ----
    def test_device_substrate_matches_host(self) -> None:
        # `ByteHashTranscript` over the `zorch.sha256` marker reproduces the host
        # `hashlib` chain exactly — the collapse's core invariant. Exercises every
        # framing branch plus a PoW grind.
        def run(byte_hash: ByteHash) -> tuple[bytes, bytes, bytes, int]:
            t = ByteHashTranscript.new(b"flock-domain", byte_hash)
            t = t.observe_label(b"phase-A").observe_bytes(b"\x00\x11\x22\x33")
            t = t.observe_scalar(b"sixteen--bytes!!")
            t = t.observe_slice(b"AABBBBCCCCCCDDDD", 4)
            t, s0 = t.sample_scalar(16)
            t, s1 = t.sample_slice(3, 16)  # 48 B — spans 2 output blocks
            t, nonce = t.observe_bytes(b"root").grind_pow(10)
            return t.buffer, s0, s1, nonce

        self.assertEqual(run(Sha256()), run(HostSha256()))


if __name__ == "__main__":
    absltest.main()
