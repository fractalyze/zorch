# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`DeviceSha256Transcript` — byte-identical to the host `Sha256Transcript`.

The host transcript is the established oracle: `byte_transcript_test` pins it to an
independent `hashlib` reference, and flock-zorch's `challenger_test` pins it to
flock-core's `FsChallenger`. This slice proves the DEVICE transcript — the same
Merlin-over-SHA-256 framing, but SHA-256 via the name-routed `zorch.sha256` marker
(`zorch.hash.sha256.digest`) instead of host `hashlib` — reproduces the host's
absorbed-byte stream and squeezed challenges exactly (fractalyze/flock-zorch#6).
"""
from __future__ import annotations

from typing import Any

from absl.testing import absltest

from zorch.byte_transcript import Sha256Transcript
from zorch.device_byte_transcript import DeviceSha256Transcript


def _run(cls: Any) -> tuple[bytes, bytes, bytes]:
    """A mixed op sequence exercising every framing branch; returns the final
    absorbed buffer plus each squeeze so a caller can compare host vs device."""
    t = cls.new(b"flock-domain")
    t = t.observe_label(b"phase-A")
    t = t.observe_bytes(b"\x00\x11\x22\x33")
    t = t.observe_scalar(b"sixteen--bytes!!")
    t = t.observe_slice(b"AABBBBCCCCCCDDDD", 4)
    t, s0 = t.sample_scalar(16)
    t, s1 = t.sample_slice(3, 16)  # 48 bytes → spans 2 SHA-256 output blocks
    return t.buffer, s0, s1


class DeviceByteTranscriptTest(absltest.TestCase):
    def test_matches_host_transcript(self) -> None:
        h_buf, h_s0, h_s1 = _run(Sha256Transcript)
        d_buf, d_s0, d_s1 = _run(DeviceSha256Transcript)
        self.assertEqual(d_s0, h_s0)
        self.assertEqual(d_s1, h_s1)
        self.assertEqual(d_buf, h_buf)

    def test_multiblock_squeeze_matches(self) -> None:
        # A squeeze spanning many SHA-256 output blocks (counter mode, 160 B = 5).
        h = Sha256Transcript.new(b"d").sample_slice(10, 16)[1]
        d = DeviceSha256Transcript.new(b"d").sample_slice(10, 16)[1]
        self.assertEqual(d, h)

    def test_domain_separation_matches(self) -> None:
        for dom in (b"dom-a", b"dom-b", b""):
            h = Sha256Transcript.new(dom).sample_scalar(16)[1]
            d = DeviceSha256Transcript.new(dom).sample_scalar(16)[1]
            self.assertEqual(d, h)

    def test_reabsorb_advances_like_host(self) -> None:
        # sample-then-observe must diverge from the no-sample path, identically on
        # both — pins the re-absorb of squeezed bytes.
        h = Sha256Transcript.new(b"d").sample_scalar(16)[0].observe_scalar(b"Z" * 16)
        d = (
            DeviceSha256Transcript.new(b"d")
            .sample_scalar(16)[0]
            .observe_scalar(b"Z" * 16)
        )
        self.assertEqual(d.sample_scalar(16)[1], h.sample_scalar(16)[1])


if __name__ == "__main__":
    absltest.main()
